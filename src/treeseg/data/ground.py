"""
ground.py - run CSF ground removal and show what it did, so it can be judged by eye.

The true labels sort every point into one of three classes:
  REMOVED: CSF called it ground
  KEPT_NOT_TREE: CSF kept it, but it is not a labelled tree (ground it missed, low plants, understory)
  KEPT_TREE: CSF kept it, and it is a labelled tree
"""
import CSF
import laspy
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import minimum_filter

REMOVED, KEPT_NOT_TREE, KEPT_TREE = 1, 2, 3
CLASS_STYLE = {KEPT_TREE: ("#c3c2b7", "Kept: tree"), KEPT_NOT_TREE: ("#eb6834", "Kept: not a tree"),
               REMOVED: ("#2a78d6", "Removed as ground")}


def load_tile(tile_dir: str, tile: str) -> tuple[np.ndarray, np.ndarray]:
    las = laspy.read(f"{tile_dir}/{tile}_scan.laz")
    return np.column_stack([las.x, las.y, las.z]), np.load(f"{tile_dir}/{tile}_instances.npy")


def csf_ground_mask(xyz: np.ndarray, resolution: float = 2.0, threshold: float = 0.5, iterations: int = 500,
                    rigidness: int = 3) -> np.ndarray:
    """True for points CSF calls ground. The defaults are the settings Kevin's benchmark uses.
    Lower rigidness lets the simulated cloth bend more, and a smaller resolution makes it finer."""
    csf = CSF.CSF()
    csf.params.cloth_resolution = resolution
    csf.params.class_threshold = threshold
    csf.params.interations = iterations  # the library spells it this way
    csf.params.rigidness = rigidness
    csf.setPointCloud(xyz)
    ground, other = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, other, exportCloth=False)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.array(ground, dtype=np.int64)] = True
    return mask


def classify(ground: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.where(ground, REMOVED, np.where(ids > 0, KEPT_TREE, KEPT_NOT_TREE)).astype(np.uint8)


def height_above_ground(xyz: np.ndarray, cell: float = 1.0, window: int = 5) -> np.ndarray:
    """Rough height: distance above the lowest point in the surrounding few metres."""
    ij = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / cell).astype(int)
    grid = np.full(ij.max(axis=0) + 1, np.inf)
    np.minimum.at(grid, (ij[:, 0], ij[:, 1]), xyz[:, 2])
    grid = minimum_filter(grid, size=window)
    return xyz[:, 2] - grid[ij[:, 0], ij[:, 1]]


def summary(xyz: np.ndarray, ids: np.ndarray, classes: np.ndarray) -> dict:
    """Counts behind the pictures, including how high the leftover non-tree points sit."""
    kept = classes == KEPT_NOT_TREE
    height = height_above_ground(xyz)
    bands = {f"{lo}-{hi} m": float(((height >= lo) & (height < hi))[kept].mean()) if kept.any() else 0.0
             for lo, hi in [(0, 0.3), (0.3, 1), (1, 3), (3, 8), (8, 99)]}
    return {"points": len(xyz), "removed_as_ground": float((classes == REMOVED).mean()),
            "labelled_not_tree": float((ids <= 0).mean()), "kept_not_tree_share_of_kept": float(kept.sum() / max((classes != REMOVED).sum(), 1)),
            "kept_not_tree_by_height": bands}


def plot_slice(xyz: np.ndarray, classes: np.ndarray, title: str, y: float | None = None, thickness: float = 2.0, path: str | None = None):
    """Side view of one thin slice. A clean result has blue along the bottom and no orange band there."""
    y = float(np.median(xyz[:, 1])) if y is None else y
    in_slice = np.flatnonzero(np.abs(xyz[:, 1] - y) < thickness / 2)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    for cls in (KEPT_TREE, KEPT_NOT_TREE, REMOVED):
        idx = in_slice[classes[in_slice] == cls]
        color, label = CLASS_STYLE[cls]
        ax.scatter(xyz[idx, 0], xyz[idx, 2], s=0.6 if cls == KEPT_TREE else 1.5, c=color, linewidths=0, label=f"{label} ({len(idx):,})")
    ax.set_title(f"{title}: {thickness:g} m thick side view at y = {y:.1f}", loc="left", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("height, z (m)")
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, markerscale=6, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def plot_3d(xyz: np.ndarray, classes: np.ndarray, title: str, max_points: int = 150000, seed: int = 0):
    """Interactive view you can rotate. Uses a sample, because the browser cannot draw millions of points."""
    import plotly.graph_objects as go

    rng = np.random.default_rng(seed)
    picked = rng.choice(len(xyz), min(max_points, len(xyz)), replace=False)
    fig = go.Figure()
    for cls in (KEPT_TREE, KEPT_NOT_TREE, REMOVED):
        idx = picked[classes[picked] == cls]
        color, label = CLASS_STYLE[cls]
        fig.add_trace(go.Scatter3d(x=xyz[idx, 0], y=xyz[idx, 1], z=xyz[idx, 2], mode="markers", name=label,
                                   marker={"size": 1.5, "color": color}))
    fig.update_layout(title=title, height=650, scene={"aspectmode": "data"}, legend={"itemsizing": "constant"})
    return fig


def write_for_viewers(path_prefix: str, xyz: np.ndarray, classes: np.ndarray) -> None:
    """Write files for other tools: _csf.laz for CloudCompare (colour by csf_class), and the annotator's trio."""
    import json
    import os

    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="csf_class", type=np.uint8))
    las = laspy.LasData(header)
    las.x, las.y, las.z, las.csf_class = xyz[:, 0], xyz[:, 1], xyz[:, 2], classes
    las.write(path_prefix + "_csf.laz")
    las.write(path_prefix + "_scan.laz")
    np.save(path_prefix + "_instances.npy", classes.astype(np.int32))
    with open(path_prefix + "_meta.json", "w") as f:
        json.dump({"tile_name": os.path.basename(path_prefix), "classes": {str(k): v[1] for k, v in CLASS_STYLE.items()}}, f, indent=2)
