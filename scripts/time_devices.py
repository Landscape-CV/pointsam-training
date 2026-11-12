"""Time Point-SAM on the devices this machine has (cpu, mps, cuda), on one 12 m crop of a tile.

It also prints how far the masks from each device agree with the first device's, as a check that the
devices compute the same thing. Clicks come from the true labels, one per tree.

Usage, from the repo root:
  python scripts/time_devices.py --checkpoint <model.safetensors> [--tile isprs_val_t00_00] [--whole-tile mps]
"""
import argparse
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, "src")

from treeseg.data.ground import csf_ground_mask, load_tile
from treeseg.engine.clicks import perfect_clicks
from treeseg.engine.segment import click_candidates, fuse_masks, predict_click_masks
from treeseg.models.point_sam import build_point_sam
from treeseg.models.torkit3d_fallback import install_torkit3d


def wait(device: str) -> None:
    """GPU work is queued, so the clock is only right once the queue is empty."""
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/experiment/diagnose_val.yaml")
    parser.add_argument("--tile", default=None, help="default: the config's first tile")
    parser.add_argument("--clicks", type=int, default=3, help="how many single-click crops to time per device")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--whole-tile", default=None, metavar="DEVICE", help="also time every click of the tile on this device")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tile = args.tile or cfg["tiles"][0]
    devices = args.devices or (["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
                               + (["cuda"] if torch.cuda.is_available() else []))
    xyz, ids = load_tile(cfg["tile_dir"], tile)
    keep = ~csf_ground_mask(xyz, **cfg["csf"])
    xyz, ids = xyz[keep], ids[keep]
    groups = perfect_clicks(xyz, ids)
    print(f"{tile}: {len(xyz):,} points after ground removal, {len(groups)} label clicks, torkit3d: {install_torkit3d()}")
    print(f"torch {torch.__version__}, crop {cfg['crop_size']} m, {cfg['crop_points']} points, devices {devices}")

    reference = None
    for device in devices:
        model = build_point_sam(args.checkpoint, cfg["group_number"], cfg["group_size"]).to(device).eval()
        click_candidates(model, xyz, groups[0], cfg["crop_size"], cfg["crop_points"], device)  # warm-up, not timed
        wait(device)
        times, results = [], []
        for group in groups[:args.clicks]:
            for _ in range(args.repeats):
                start = time.perf_counter()
                out = click_candidates(model, xyz, group, cfg["crop_size"], cfg["crop_points"], device)
                wait(device)
                times.append(time.perf_counter() - start)
            results.append(out)
        line = f"{device}: {np.median(times):.2f} s per crop (median of {len(times)}, range {min(times):.2f} to {max(times):.2f})"
        if reference is None:
            reference = results
        else:
            #the model's own pick on each device, compared point for point
            ious = []
            for a, b in zip(reference, results):
                ma, mb = a["masks"][a["picked"]], b["masks"][b["picked"]]
                ious.append((ma & mb).sum() / max(1, (ma | mb).sum()))
            line += f", picked-mask IoU against {devices[0]}: {np.round(ious, 4).tolist()}"
        print(line, flush=True)

        if args.whole_tile == device:
            start = time.perf_counter()
            raw = predict_click_masks(model, xyz, groups, crop_size=cfg["crop_size"], crop_stride=cfg["crop_stride"],
                                      crop_points=cfg["crop_points"], device=device)
            wait(device)
            seconds = time.perf_counter() - start
            labels = fuse_masks(len(xyz), *raw)
            print(f"{device}: whole tile, {len(groups)} clicks in {seconds:.1f} s, {labels.max() + 1} trees after merging")
        del model


if __name__ == "__main__":
    main()
