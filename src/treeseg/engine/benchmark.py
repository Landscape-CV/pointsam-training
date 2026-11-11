"""
benchmark.py - score checkpoints on whole tiles: ground removal, clicks, masks, fusion, F1.
Each click method is scored before and after merging, next to baselines that need no model.

Written fresh, so it has no dependency on Ecomodel. Rows go to a CSV as they finish.
"""
import csv
import os
import time

import numpy as np
import pandas as pd

from ..data.ground import csf_ground_mask, load_tile
from ..models.point_sam import build_point_sam
from .clicks import clicks_at_positions, make_clicks, treex_segment
from .scoring import click_quality, score_instances, score_raw_masks, tree_sizes
from .segment import fuse_masks, predict_click_masks, save_masks


def prepare_tile(tile_dir: str, tile: str, csf: dict, min_tree_points: int = 200, keep: np.ndarray | None = None) -> dict:
    """Load a tile and remove the ground. Tree sizes come from before removal, so lost trees still count.
    Pass keep (from load_masks) to reuse a saved ground removal, which saved masks and predictions line up with."""
    xyz, ids = load_tile(tile_dir, tile)
    sizes = tree_sizes(ids, min_tree_points)
    keep = ~csf_ground_mask(xyz, **csf) if keep is None else keep
    return {"name": tile, "xyz": xyz[keep], "ids": ids[keep], "sizes": sizes, "full_xyz": xyz, "keep": keep}


def treex_full(tile: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """Baseline with no model: TreeX's own tree labels, on the points left after ground removal."""
    labels, positions = treex_segment(tile["full_xyz"])
    return np.where(labels >= 0, labels, -1)[tile["keep"]], clicks_at_positions(tile["xyz"], positions)


#methods that segment a tile by themselves. Each returns a label per kept point and the clicks its trunks would give.
BASELINES = {"treex_full": treex_full}
NO_RAW_MASKS = {"raw_masks": float("nan"), "raw_trees_found": float("nan"), "raw_recall": float("nan"), "raw_mean_tree_iou": float("nan")}


def run_benchmark(cfg: dict, device: str = "cuda") -> str:
    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], "results.csv")
    tiles = [prepare_tile(cfg["tile_dir"], name, cfg["csf"], cfg["min_tree_points"]) for name in cfg["tiles"]]
    rows = []

    def record(checkpoint: str, tile: dict, method: str, groups: list, pred: np.ndarray, raw: dict, start: float) -> None:
        base = {"checkpoint": checkpoint, "tile": tile["name"], "clicks_from": method,
                **click_quality(tile["xyz"], tile["ids"], groups, tile["sizes"]), **raw}
        #scored twice: strict, and without penalising predictions on unlabelled understory
        for rule, ignore in (("strict", False), ("ignore_non_tree", True)):
            row = {**base, "rule": rule, **score_instances(tile["ids"], pred, tile["sizes"], cfg["iou_threshold"], ignore),
                   "seconds": round(time.time() - start, 1)}
            rows.append(row)
            print(f"{checkpoint} | {tile['name']} | {method} | {rule}: F1={row['f1']:.3f} P={row['precision']:.3f} "
                  f"R={row['recall']:.3f} raw recall={row['raw_recall']:.3f} click recall={row['click_recall']:.2f}", flush=True)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        np.save(os.path.join(cfg["out_dir"], f"{checkpoint}_{tile['name']}_{method}_pred.npy"), pred)

    for name in cfg.get("baselines", []):
        for tile in tiles:
            start = time.time()
            pred, groups = BASELINES[name](tile)
            record(name, tile, "none", groups, pred, NO_RAW_MASKS, start)

    clicks = {(t["name"], m): make_clicks(m, t["xyz"], t["ids"], t["full_xyz"]) for t in tiles for m in cfg["click_methods"]}
    #negative_clicks is a list, so one run can compare none against some. A count above 0 is added to the checkpoint's name
    negative_counts = cfg.get("negative_clicks", [0])
    for checkpoint, checkpoint_path in cfg["checkpoints"].items():
        model = build_point_sam(checkpoint_path, cfg["group_number"], cfg["group_size"]).to(device).eval()
        for negatives in negative_counts:
            label = checkpoint if not negatives else f"{checkpoint}_neg{negatives}"
            for tile in tiles:
                for method in cfg["click_methods"]:
                    start = time.time()
                    groups = clicks[(tile["name"], method)]
                    raw_masks = predict_click_masks(model, tile["xyz"], groups, crop_size=cfg["crop_size"], crop_stride=cfg["crop_stride"],
                                                    crop_points=cfg["crop_points"], device=device, negative_clicks=negatives,
                                                    negative_radius=cfg.get("negative_radius", 6.0))
                    pred = fuse_masks(len(tile["xyz"]), *raw_masks)
                    #the same masks scored before merging, to separate mask quality from what merging does to it
                    raw = score_raw_masks(tile["ids"], raw_masks[0], tile["sizes"], cfg["iou_threshold"])
                    del raw["best"]
                    record(label, tile, method, groups, pred, raw, start)
                    if cfg.get("save_masks"):
                        save_masks(os.path.join(cfg["out_dir"], f"{label}_{tile['name']}_{method}_masks.npz"), groups, *raw_masks, keep=tile["keep"])
        del model
    return path


def summarise(path: str) -> pd.DataFrame:
    """Average over tiles, with one row per checkpoint, click source and scoring rule."""
    table = pd.read_csv(path)
    columns = ["f1", "precision", "recall", "raw_recall", "mean_tree_iou", "raw_mean_tree_iou", "mean_matched_iou", "click_recall", "clicks_on_tree"]
    #results saved before raw masks were scored don't have the raw columns
    return table.groupby(["checkpoint", "clicks_from", "rule"], as_index=False)[[c for c in columns if c in table]].mean().round(3)
