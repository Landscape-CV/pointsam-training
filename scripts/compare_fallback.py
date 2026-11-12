"""Point-SAM with the real torkit3d against the plain-PyTorch fallback: same masks, same scores, same speed?

Run this on a CUDA machine (ICE). Both paths run on the same GPU, in the same process, on the same points and
the same clicks, so any difference is down to the three replaced functions and nothing else. It reports:

  1. masks     per single-click crop: IoU between the two paths' masks, and whether farthest point sampling
               picked the same points
  2. speed     seconds per crop, seconds for point sampling alone, seconds for a whole tile
  3. scores    whole-tile F1, precision, recall and raw recall for each click method, and the val-crop
               scores (val_iou_pick and the rest), each path next to the other

A Mac cannot run the real torkit3d. There, --dry-run runs the fallback against itself, which only proves the
script works: its differences are 0 by construction and mean nothing.

Usage, from the repo root, inside a GPU job (or: sbatch scripts/slurm/compare_fallback.sbatch <checkpoint>):
  python scripts/compare_fallback.py --checkpoint results/isprs_lora/best.safetensors
"""
import argparse
import contextlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "src")

from torch.utils.data import DataLoader, Subset

from treeseg.data.dataset import ForestCrops
from treeseg.engine.benchmark import prepare_tile
from treeseg.engine.clicks import make_clicks
from treeseg.engine.scoring import score_instances, score_raw_masks
from treeseg.engine.segment import click_candidates, fuse_masks, predict_click_masks
from treeseg.engine.trainer import collate, validate
from treeseg.models import torkit3d_fallback as fallback
from treeseg.models.point_sam import build_point_sam

PATHS = ("torkit3d", "fallback")
TILE_SCORES = ["f1", "precision", "recall", "raw_recall", "mean_tree_iou", "raw_mean_tree_iou"]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def wait(device: str) -> None:
    """GPU work is queued, so the clock is only right once the queue is empty."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def timed(fn, device: str):
    wait(device)
    start = time.perf_counter()
    out = fn()
    wait(device)
    return out, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiment/diagnose_val.yaml")
    parser.add_argument("--tiles", type=int, default=None, help="how many of the config's tiles (default: all)")
    parser.add_argument("--click-methods", nargs="+", default=["perfect", "stems"])
    parser.add_argument("--clicks", type=int, default=10, help="single-click crops for the mask comparison")
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per crop")
    parser.add_argument("--val-crops", type=int, default=None, help="how many val crops to score (default: all)")
    parser.add_argument("--out-dir", default="results/compare_fallback")
    parser.add_argument("--dry-run", action="store_true", help="no CUDA needed: fallback against itself, to test this script only")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.dry_run:
        device = fallback.pick_device()
        print(f"DRY RUN on {device}: both columns are the fallback. This checks the script, not the fallback.")
    elif fallback.real_torkit3d_usable():
        device = "cuda"
    else:
        sys.exit("The real torkit3d cannot run here (no CUDA, or it is not installed). Use --dry-run to test the script only.")

    model = build_point_sam(args.checkpoint, cfg["group_number"], cfg["group_size"]).to(device).eval()
    if args.dry_run:
        real_fps = fallback.sample_farthest_points
        real_chamfer = fallback.chamfer_distance
    else:
        from torkit3d.ops.chamfer_distance import chamfer_distance as real_chamfer
        from torkit3d.ops.sample_farthest_points import sample_farthest_points as real_fps

    def path(name: str):
        """Inside this block the model uses the named path."""
        return fallback.use_fallback() if name == "fallback" else contextlib.nullcontext()

    os.makedirs(args.out_dir, exist_ok=True)
    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"device: {name}, torch {torch.__version__}, checkpoint {args.checkpoint}")
    tiles = [prepare_tile(cfg["tile_dir"], tile, cfg["csf"], cfg["min_tree_points"]) for tile in cfg["tiles"][:args.tiles]]
    crop = dict(crop_size=cfg["crop_size"], crop_points=cfg["crop_points"], device=device)

    # 1 and 2a: single-click crops, masks compared point for point, and timed
    tile = tiles[0]
    groups = make_clicks("perfect", tile["xyz"], tile["ids"], tile["full_xyz"])[:args.clicks]
    with path("torkit3d"):
        click_candidates(model, tile["xyz"], groups[0], **crop)  # warm-up, not timed
    rows = []
    for number, group in enumerate(groups):
        out, seconds = {}, {}
        for p in PATHS:
            with path(p):
                runs = [timed(lambda: click_candidates(model, tile["xyz"], group, **crop), device) for _ in range(args.repeats)]
            out[p], seconds[p] = runs[-1][0], float(np.median([s for _, s in runs]))
        real, ours = out["torkit3d"], out["fallback"]
        assert np.array_equal(real["sample_index"], ours["sample_index"]), "the two paths saw different points"

        #the same crop, normalised as click_candidates does it, for the two functions on their own
        points = tile["xyz"][real["sample_index"]]
        centred = points - points.mean(axis=0)
        coords = torch.from_numpy((centred / (np.linalg.norm(centred, axis=1).max() or 1.0)).astype(np.float32))[None].to(device)
        real_idx, fps_real_s = timed(lambda: real_fps(coords, cfg["group_number"]), device)
        our_idx, fps_ours_s = timed(lambda: fallback.sample_farthest_points(coords, cfg["group_number"]), device)
        #a mask-shaped split, as the click sampler in training uses chamfer
        inside = torch.from_numpy(real["masks"][real["picked"]]).to(device)
        chamfer_diff = float("nan")
        if inside.any() and not inside.all():
            a, b = coords[:, inside], coords[:, ~inside]
            chamfer_diff = max(float((x - y).abs().max()) for x, y in zip(real_chamfer(a, b), fallback.chamfer_distance(a, b)))
        rows.append({"click": number,
                     "picked_mask_iou": iou(real["masks"][real["picked"]], ours["masks"][ours["picked"]]),
                     "same_pick": real["picked"] == ours["picked"],
                     "worst_iou_of_3": min(iou(a, b) for a, b in zip(real["masks"], ours["masks"])),
                     "fps_picks_equal": float((real_idx == our_idx).float().mean()),
                     "fps_same_set": len(set(real_idx[0].tolist()) & set(our_idx[0].tolist())) / cfg["group_number"],
                     "chamfer_max_diff": chamfer_diff,
                     "crop_s_torkit3d": seconds["torkit3d"], "crop_s_fallback": seconds["fallback"],
                     "fps_s_torkit3d": fps_real_s, "fps_s_fallback": fps_ours_s})
    crops = pd.DataFrame(rows)
    crops.to_csv(os.path.join(args.out_dir, "crops.csv"), index=False)
    print(f"\n1. MASKS, {len(crops)} single-click crops of {tile['name']} (1.0 = identical)")
    print(crops[["click", "picked_mask_iou", "same_pick", "worst_iou_of_3", "fps_picks_equal", "fps_same_set", "chamfer_max_diff"]]
          .round(4).to_string(index=False))
    print(f"picked-mask IoU between the paths: mean {crops.picked_mask_iou.mean():.4f}, lowest {crops.picked_mask_iou.min():.4f}")

    # 2b and 3a: whole tiles, the same clicks for both paths
    rows = []
    for tile in tiles:
        for method in args.click_methods:
            groups = make_clicks(method, tile["xyz"], tile["ids"], tile["full_xyz"])
            preds = {}
            for p in PATHS:
                with path(p):
                    raw, seconds = timed(lambda: predict_click_masks(model, tile["xyz"], groups, crop_stride=cfg["crop_stride"], **crop), device)
                preds[p] = fuse_masks(len(tile["xyz"]), *raw)
                raw_scores = score_raw_masks(tile["ids"], raw[0], tile["sizes"], cfg["iou_threshold"])
                scores = {**score_instances(tile["ids"], preds[p], tile["sizes"], cfg["iou_threshold"], True), **raw_scores}
                rows.append({"tile": tile["name"], "clicks_from": method, "path": p, "clicks": len(groups), "seconds": seconds,
                             **{key: scores[key] for key in TILE_SCORES}})
            #share of points given the same answer (tree or no tree is compared, since tree numbers are arbitrary)
            rows[-1]["points_same_tree_or_not"] = rows[-2]["points_same_tree_or_not"] = float(((preds["torkit3d"] >= 0) == (preds["fallback"] >= 0)).mean())
    tile_table = pd.DataFrame(rows)
    tile_table.to_csv(os.path.join(args.out_dir, "tiles.csv"), index=False)
    print(f"\n3. TILE SCORES (rule ignore_non_tree, IoU {cfg['iou_threshold']}) and whole-tile seconds")
    print(tile_table.round(3).to_string(index=False))
    mean = tile_table.groupby(["clicks_from", "path"], as_index=False)[TILE_SCORES + ["seconds"]].mean()
    print("\nmean over tiles")
    print(mean.round(3).to_string(index=False))

    # 3b: val crops with label clicks, the number used to pick the checkpoint
    from pc_sam.model.loss import Criterion

    crop_set = ForestCrops(f"data/wytham_isprs/crops/{cfg.get('crop_split', 'val')}", 10000, 4, 40, seed=2)
    crop_set = Subset(crop_set, range(min(args.val_crops or len(crop_set), len(crop_set))))
    loader = DataLoader(crop_set, batch_size=1, shuffle=False, collate_fn=collate)
    #one crop untimed first, or the first path pays for start-up and looks slower
    validate(model, DataLoader(Subset(crop_set, range(1)), batch_size=1, collate_fn=collate), Criterion(use_soft_iou=False), device, "stem")
    rows = []
    for p in PATHS:
        with path(p):
            scores, seconds = timed(lambda: validate(model, loader, Criterion(use_soft_iou=False), device, "stem"), device)
        rows.append({"path": p, **scores, "seconds": seconds})
    model.eval()
    val_table = pd.DataFrame(rows)
    val_table.to_csv(os.path.join(args.out_dir, "val_crops.csv"), index=False)
    print(f"\n3. VAL CROP SCORES, {len(crop_set)} crops, label clicks")
    print(val_table.round(4).to_string(index=False))

    # 2: speed in one place
    print("\n2. SPEED, median seconds")
    whole = tile_table.groupby("path").seconds.mean()
    speed = pd.DataFrame({"what": ["one crop, one click", "point sampling alone (512 of 10,000)", "whole tile, mean", "val crops, all"],
                          "torkit3d": [crops.crop_s_torkit3d.median(), crops.fps_s_torkit3d.median(), whole["torkit3d"], val_table.seconds[0]],
                          "fallback": [crops.crop_s_fallback.median(), crops.fps_s_fallback.median(), whole["fallback"], val_table.seconds[1]]})
    speed["fallback / torkit3d"] = speed.fallback / speed.torkit3d
    speed.to_csv(os.path.join(args.out_dir, "speed.csv"), index=False)
    print(speed.round(4).to_string(index=False))
    print(f"\ntables saved in {args.out_dir}/")
    if args.dry_run:
        print("DRY RUN: both paths were the fallback, so none of the above compares it with torkit3d.")


if __name__ == "__main__":
    main()
