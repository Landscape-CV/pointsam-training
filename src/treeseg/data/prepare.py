"""
prepare.py - turn one labelled forest strip into voxels, 30 m tiles and 12 m crops.

Tree labels stay as the file's own ids (>0), and -1 means "not a tree", so a tree can be
followed across tiles and strips. Colour is not stored: the ISPRS data has none.
"""
import json
import os

import laspy
import numpy as np
from plyfile import PlyData


def load_strip(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an ISPRS PLY: points (N,3) float64 and instance ids (N,) int32."""
    v = PlyData.read(path)["vertex"].data
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    return xyz, np.asarray(v["instance"], dtype=np.int32)


def voxelize(xyz: np.ndarray, ids: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per voxel. The voxel takes its most common label, and keeps a point with it."""
    cell = np.floor((xyz - xyz.min(axis=0)) / size).astype(np.int64)
    #21 bits per axis is 2M cells, about 100 km at 5 cm
    key = (cell[:, 0] << 42) | (cell[:, 1] << 21) | cell[:, 2]

    #group points by (voxel, label): each run is one label inside one voxel
    order = np.lexsort((ids, key))
    key_s, ids_s = key[order], ids[order]
    new_run = np.r_[True, (key_s[1:] != key_s[:-1]) | (ids_s[1:] != ids_s[:-1])]
    starts = np.flatnonzero(new_run)
    counts = np.diff(np.r_[starts, len(key_s)])

    #within each voxel put the largest run first, then take the first run of each voxel
    run_key = key_s[starts]
    by_size = np.lexsort((-counts, run_key))
    first = np.r_[True, run_key[by_size][1:] != run_key[by_size][:-1]]
    keep = order[starts[by_size[first]]]
    return xyz[keep], ids[keep]


def window_starts(lo: float, hi: float, size: float, stride: float) -> list[float]:
    """Window starts along one axis. The last window sits on the far edge, so none is a thin sliver."""
    if hi - lo <= size:
        return [lo]
    starts = list(np.arange(lo, hi - size, stride))
    return starts + [hi - size]


def in_window(xyz: np.ndarray, x0: float, y0: float, size: float) -> np.ndarray:
    return (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + size) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + size)


def write_crops(xyz: np.ndarray, ids: np.ndarray, out_dir: str, name: str, size: float, stride: float,
                min_points: int, max_points: int, min_trees: int, seed: int = 0) -> list[dict]:
    """Slide a square window over the strip and save each usable window as an .npz crop."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    records = []
    for x0 in window_starts(xyz[:, 0].min(), xyz[:, 0].max(), size, stride):
        for y0 in window_starts(xyz[:, 1].min(), xyz[:, 1].max(), size, stride):
            idx = np.flatnonzero(in_window(xyz, x0, y0, size))
            n_trees = len(np.unique(ids[idx][ids[idx] > 0]))
            if len(idx) < min_points or n_trees < min_trees:
                continue
            if len(idx) > max_points:
                idx = rng.choice(idx, max_points, replace=False)
            crop_name = f"{name}_crop{len(records):03d}"
            np.savez(os.path.join(out_dir, crop_name + ".npz"),
                     xyz=xyz[idx].astype(np.float32), instance_ids=ids[idx].astype(np.int32))
            records.append({"crop": crop_name, "origin_xy": [float(x0), float(y0)],
                            "n_points": int(len(idx)), "n_trees": n_trees})
    return records


def tile_grid(xyz: np.ndarray, size: float) -> np.ndarray:
    """Tile number of each point, on the same grid as write_tiles."""
    cell = np.floor((xyz[:, :2] - xyz[:, :2].min(axis=0)) / size).astype(np.int64)
    #the far edge belongs to the last tile, as it does in write_tiles
    cell = np.minimum(cell, np.ceil(np.ptp(xyz[:, :2], axis=0) / size).astype(np.int64) - 1)
    return cell[:, 0] * 100000 + cell[:, 1]


def ground_by_tile(xyz: np.ndarray, size: float, csf: dict) -> np.ndarray:
    """Run the cloth filter one tile at a time, exactly as the benchmark does, and combine the results."""
    from .ground import csf_ground_mask

    tiles = tile_grid(xyz, size)
    ground = np.zeros(len(xyz), dtype=bool)
    for tile in np.unique(tiles):
        idx = np.flatnonzero(tiles == tile)
        ground[idx] = csf_ground_mask(xyz[idx], **csf)
    return ground


def write_tiles(xyz: np.ndarray, ids: np.ndarray, out_dir: str, name: str, size: float) -> list[dict]:
    """Cut the strip into tiles the annotator can open: _scan.laz, _instances.npy, _meta.json."""
    os.makedirs(out_dir, exist_ok=True)
    records = []
    x_min, y_min = xyz[:, 0].min(), xyz[:, 1].min()
    n_x = int(np.ceil((xyz[:, 0].max() - x_min) / size))
    n_y = int(np.ceil((xyz[:, 1].max() - y_min) / size))
    for ix in range(n_x):
        for iy in range(n_y):
            x0, y0 = x_min + ix * size, y_min + iy * size
            #the last row and column include the far edge, which `<` would otherwise drop
            x_hi = x0 + size if ix < n_x - 1 else np.inf
            y_hi = y0 + size if iy < n_y - 1 else np.inf
            m = (xyz[:, 0] >= x0) & (xyz[:, 0] < x_hi) & (xyz[:, 1] >= y0) & (xyz[:, 1] < y_hi)
            if not m.any():
                continue
            tile = f"{name}_t{ix:02d}_{iy:02d}"
            prefix = os.path.join(out_dir, tile)
            header = laspy.LasHeader(point_format=3, version="1.2")
            header.offsets = xyz[m].min(axis=0)
            header.scales = [0.001, 0.001, 0.001]
            las = laspy.LasData(header)
            las.x, las.y, las.z = xyz[m, 0], xyz[m, 1], xyz[m, 2]
            las.write(prefix + "_scan.laz")
            np.save(prefix + "_instances.npy", ids[m].astype(np.int32))
            n_trees = len(np.unique(ids[m][ids[m] > 0]))
            record = {"tile_name": tile, "strip": name, "num_points": int(m.sum()), "num_trees": n_trees,
                      "label_schema": {"not_tree": -1, "trees": ">0"}}
            with open(prefix + "_meta.json", "w") as f:
                json.dump(record, f, indent=2)
            records.append(record)
    return records
