"""
overview.py - tables, pictures and checks of the built dataset, for notebooks/00_dataset.ipynb.

Everything here reads the tiles, crops and summary.json on disk, so it works the same whether they
were just built or downloaded.
"""
import glob
import json
import os

import laspy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ..utils.view3d import colour_by_id

NOT_TREE = "#c9c8c0"
SPLIT_COLOUR = {"train": "#2a78d6", "val": "#eda100", "test": "#e34948"}


def tile_table(tile_dir: str) -> list[dict]:
    """One row per tile, from its _meta.json."""
    rows = []
    for path in sorted(glob.glob(os.path.join(tile_dir, "*_meta.json"))):
        meta = json.load(open(path))
        rows.append({"tile": meta["tile_name"], "strip": meta["strip"].replace("isprs_", ""), "points": meta["num_points"], "trees": meta["num_trees"]})
    return rows


def load_overview(tile_dir: str, per_tile: int = 60000, seed: int = 0) -> dict:
    """A thinned copy of every tile for drawing, and each tile's outline. Keyed by tile name."""
    rng = np.random.default_rng(seed)
    tiles = {}
    for row in tile_table(tile_dir):
        las = laspy.read(os.path.join(tile_dir, row["tile"] + "_scan.laz"))
        xyz = np.column_stack([las.x, las.y, las.z])
        ids = np.load(os.path.join(tile_dir, row["tile"] + "_instances.npy"))
        pick = rng.choice(len(xyz), min(per_tile, len(xyz)), replace=False)
        tiles[row["tile"]] = {"strip": row["strip"], "xyz": xyz[pick], "ids": ids[pick], "box": (*xyz[:, :2].min(axis=0), *xyz[:, :2].max(axis=0))}
    return tiles


def trees_by_strip(tile_dir: str) -> dict[str, dict[int, int]]:
    """Every tree in each strip and its point count, added up over the strip's tiles."""
    strips: dict[str, dict[int, int]] = {}
    for row in tile_table(tile_dir):
        ids = np.load(os.path.join(tile_dir, row["tile"] + "_instances.npy"))
        trees, counts = np.unique(ids[ids > 0], return_counts=True)
        mine = strips.setdefault(row["strip"], {})
        for tree, count in zip(trees.tolist(), counts.tolist()):
            mine[tree] = mine.get(tree, 0) + count
    return strips


def split_checks(tile_dir: str, tiles: dict) -> list[dict]:
    """The claims the split rests on, counted from the labels. tiles is what load_overview returned."""
    trees = trees_by_strip(tile_dir)
    boxes = {}
    for tile in tiles.values():
        boxes.setdefault(tile["strip"], []).append(tile["box"])
    span = {s: (min(b[0] for b in bs), max(b[2] for b in bs)) for s, bs in boxes.items()}
    gap = max(span["train"][0] - span["test"][1], span["test"][0] - span["train"][1])
    shared = lambda a, b: len(set(trees[a]) & set(trees[b]))
    return [{"check": "trees in train", "value": len(trees["train"])}, {"check": "trees in val", "value": len(trees["val"])},
            {"check": "trees in test", "value": len(trees["test"])},
            {"check": "trees in both train and test (must be 0)", "value": shared("train", "test")},
            {"check": "trees in both val and train", "value": shared("val", "train")},
            {"check": "trees in both val and test", "value": shared("val", "test")},
            {"check": "gap between train and test strips, metres", "value": f"{gap:.1f}"}]


def plot_strips(tiles: dict, title: str = ""):
    """The whole scan from above, each tree in its own colour, with tile outlines."""
    fig, ax = plt.subplots(figsize=(15, 9))
    for name, tile in tiles.items():
        tree = tile["ids"] > 0
        ax.scatter(tile["xyz"][~tree, 0], tile["xyz"][~tree, 1], s=0.15, c=NOT_TREE, linewidths=0, rasterized=True)
        ax.scatter(tile["xyz"][tree, 0], tile["xyz"][tree, 1], s=0.25, c=colour_by_id(tile["ids"][tree]), linewidths=0, rasterized=True)
    for name, tile in tiles.items():
        x0, y0, x1, y1 = tile["box"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=SPLIT_COLOUR[tile["strip"]], linewidth=2))
        ax.text(x0 + 0.6, y1 - 0.6, name.replace("isprs_", ""), fontsize=8, va="top", color="black",
                bbox={"facecolor": "white", "edgecolor": SPLIT_COLOUR[tile["strip"]], "pad": 1.5})
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or "Wytham from above. Outline colour: blue train, yellow val, red test. Tree colours repeat and are for display only", loc="left")
    return fig


def plot_crop_windows(tiles: dict, strip: str, crops: list[dict], crop_m: float, highlight: int = 0):
    """Where the crops of one strip sit. They overlap by half, so the outlines form a grid finer than the crop."""
    fig, ax = plt.subplots(figsize=(11, 9))
    for tile in tiles.values():
        if tile["strip"] == strip:
            ax.scatter(tile["xyz"][:, 0], tile["xyz"][:, 1], s=0.2, c=np.where(tile["ids"] > 0, "#8a8a80", NOT_TREE), linewidths=0, rasterized=True)
    for k, crop in enumerate(crops):
        x0, y0 = crop["origin_xy"]
        ax.add_patch(Rectangle((x0, y0), crop_m, crop_m, fill=k == highlight, facecolor="#eda10055", edgecolor="#e34948" if k == highlight else "#2a78d6",
                               linewidth=2 if k == highlight else 0.5, zorder=3 if k == highlight else 2))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{strip} strip: {len(crops)} crops of {crop_m:g} m. Crop {highlight} is filled in", loc="left")
    return fig


def plot_crops(crop_dir: str, numbers: list[int]):
    """A few crops, each from above and from the side, coloured by tree."""
    paths = sorted(glob.glob(os.path.join(crop_dir, "*.npz")))
    fig, axes = plt.subplots(2, len(numbers), figsize=(4.2 * len(numbers), 8.4), squeeze=False)
    for column, number in enumerate(numbers):
        crop = np.load(paths[number])
        xyz, ids = crop["xyz"], crop["instance_ids"]
        colours = np.where(ids > 0, np.array(colour_by_id(ids)), NOT_TREE)
        for row, (u, v, view) in enumerate(((0, 1, "from above"), (0, 2, "from the side"))):
            ax = axes[row, column]
            ax.scatter(xyz[:, u], xyz[:, v], s=0.3, c=colours, linewidths=0, rasterized=True)
            ax.set_aspect("equal")
            ax.set_title(f"crop {number}, {view}: {len(xyz):,} points, {len(np.unique(ids[ids > 0]))} trees" if row == 0 else view, fontsize=9, loc="left")
    return fig


def plot_ground_removed(tile_dir: str, crop_path: str, origin_xy: list[float], crop_m: float):
    """The same 12 m window twice, from the side: as it is in the tiles, and as the crop the model trains on."""
    x0, y0 = origin_xy
    window = []
    for row in tile_table(tile_dir):
        las = laspy.read(os.path.join(tile_dir, row["tile"] + "_scan.laz"))
        xyz = np.column_stack([las.x, las.y, las.z])
        inside = (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + crop_m) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + crop_m)
        if inside.any():
            window.append((xyz[inside], np.load(os.path.join(tile_dir, row["tile"] + "_instances.npy"))[inside]))
    tile_xyz, tile_ids = np.vstack([w[0] for w in window]), np.concatenate([w[1] for w in window])
    crop = np.load(crop_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, xyz, ids, title in ((axes[0], tile_xyz, tile_ids, "In the tiles: every point, ground included"),
                                (axes[1], crop["xyz"], crop["instance_ids"], "The crop: ground removed, at most 100,000 points")):
        ax.scatter(xyz[:, 0], xyz[:, 2], s=0.2, c=np.where(ids > 0, np.array(colour_by_id(ids)), "#55554f"), linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(f"{title} ({len(xyz):,} points)", loc="left", fontsize=10)
    axes[0].set_ylabel("height (m)")
    return fig
