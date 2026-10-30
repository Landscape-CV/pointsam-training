"""
benchmark.py - score checkpoints on whole tiles: ground removal, clicks, masks, fusion, F1.

Written fresh, so it has no dependency on Ecomodel. Rows go to a CSV as they finish.
"""
import csv
import os
import time

import numpy as np
import pandas as pd

from ..data.ground import csf_ground_mask, load_tile
from ..models.point_sam import build_point_sam
from .clicks import grid_clicks, perfect_clicks, stem_clicks
from .scoring import click_quality, score_instances, tree_sizes
from .segment import segment_tile


def prepare_tile(tile_dir: str, tile: str, csf: dict, min_tree_points: int = 200) -> dict:
    """Load a tile and remove the ground. Tree sizes come from before removal, so lost trees still count."""
    xyz, ids = load_tile(tile_dir, tile)
    sizes = tree_sizes(ids, min_tree_points)
    keep = ~csf_ground_mask(xyz, **csf)
    return {"name": tile, "xyz": xyz[keep], "ids": ids[keep], "sizes": sizes}


def make_clicks(method: str, xyz: np.ndarray, ids: np.ndarray) -> list[np.ndarray]:
    if method == "grid":
        return grid_clicks(xyz)
    if method == "stems":
        return stem_clicks(xyz)
    if method == "perfect":
        return perfect_clicks(xyz, ids)
    raise ValueError(f"unknown click method: {method}")


def run_benchmark(cfg: dict, device: str = "cuda") -> str:
    os.makedirs(cfg["out_dir"], exist_ok=True)
    path = os.path.join(cfg["out_dir"], "results.csv")
    tiles = [prepare_tile(cfg["tile_dir"], name, cfg["csf"], cfg["min_tree_points"]) for name in cfg["tiles"]]
    clicks = {(t["name"], m): make_clicks(m, t["xyz"], t["ids"]) for t in tiles for m in cfg["click_methods"]}

    rows = []
    for checkpoint, checkpoint_path in cfg["checkpoints"].items():
        model = build_point_sam(checkpoint_path, cfg["group_number"], cfg["group_size"]).to(device).eval()
        for tile in tiles:
            for method in cfg["click_methods"]:
                start = time.time()
                groups = clicks[(tile["name"], method)]
                pred = segment_tile(model, tile["xyz"], groups, device, crop_size=cfg["crop_size"],
                                    crop_stride=cfg["crop_stride"], crop_points=cfg["crop_points"])
                base = {"checkpoint": checkpoint, "tile": tile["name"], "clicks_from": method,
                        **click_quality(tile["xyz"], tile["ids"], groups, tile["sizes"])}
                #scored twice: strict, and without penalising predictions on unlabelled understory
                for rule, ignore in (("strict", False), ("ignore_non_tree", True)):
                    row = {**base, "rule": rule, **score_instances(tile["ids"], pred, tile["sizes"], cfg["iou_threshold"], ignore),
                           "seconds": round(time.time() - start, 1)}
                    rows.append(row)
                    print(f"{checkpoint} | {tile['name']} | {method} | {rule}: F1={row['f1']:.3f} P={row['precision']:.3f} "
                          f"R={row['recall']:.3f} click recall={row['click_recall']:.2f}", flush=True)
                    with open(path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                np.save(os.path.join(cfg["out_dir"], f"{checkpoint}_{tile['name']}_{method}_pred.npy"), pred)
        del model
    return path


def summarise(path: str) -> pd.DataFrame:
    """Average over tiles, with one row per checkpoint, click source and scoring rule."""
    table = pd.read_csv(path)
    columns = ["f1", "precision", "recall", "mean_matched_iou", "click_recall", "clicks_on_tree"]
    return table.groupby(["checkpoint", "clicks_from", "rule"], as_index=False)[columns].mean().round(3)
