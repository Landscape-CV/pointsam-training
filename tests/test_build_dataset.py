"""build_dataset on a tiny made-up strip: the same call the script and notebook 00 make."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

pytest.importorskip("CSF")
from plyfile import PlyData, PlyElement

from treeseg.data.download import PARTS
from treeseg.data.prepare import build_dataset


def write_strip(path: str, seed: int) -> None:
    """A flat ground and four upright sticks, each its own tree."""
    rng = np.random.default_rng(seed)
    ground = np.column_stack([rng.uniform(0, 8, 4000), rng.uniform(0, 8, 4000), rng.normal(0, 0.01, 4000)])
    trees, ids = [], []
    for tree, (x, y) in enumerate(((2, 2), (6, 2), (2, 6), (6, 6)), start=1):
        trees.append(np.column_stack([x + rng.normal(0, 0.1, 1500), y + rng.normal(0, 0.1, 1500), rng.uniform(0.5, 5, 1500)]))
        ids.append(np.full(1500, tree))
    xyz = np.vstack([ground, *trees])
    labels = np.concatenate([np.full(len(ground), -1), *ids])
    vertex = np.zeros(len(xyz), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("instance", "i4")])
    vertex["x"], vertex["y"], vertex["z"], vertex["instance"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], labels
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def test_build_dataset_writes_tiles_crops_and_summary(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write_strip(str(raw / "a.ply"), 0)
    write_strip(str(raw / "b.ply"), 1)
    cfg = {"source_dir": str(raw), "out_dir": str(out), "strips": {"train": "a.ply", "val": "b.ply"}, "voxel_m": 0.05, "tile_m": 30.0,
           "crop_m": 4.0, "stride_m": 2.0, "min_crop_points": 100, "max_crop_points": 2000, "min_crop_trees": 1, "seed": 0,
           "csf": {"resolution": 0.5, "threshold": 0.5, "iterations": 200, "rigidness": 1}}
    summary = build_dataset(cfg)

    assert json.load(open(out / "summary.json"))["strips"].keys() == {"train", "val"}
    for split in ("train", "val"):
        row = summary["strips"][split]
        assert row["voxels"] < row["raw_points"]
        assert os.path.exists(out / "tiles" / f"isprs_{split}_t00_00_scan.laz")
        labels = np.load(out / "tiles" / f"isprs_{split}_t00_00_instances.npy")
        assert len(labels) == row["voxels"] and set(np.unique(labels)) == {-1, 1, 2, 3, 4}
        files = sorted(f for f in os.listdir(out / "crops" / split) if f.endswith(".npz"))
        assert len(files) == len(row["crops"]) > 0
        #each crop also lists the trees that stand inside its window, and every such tree is in the crop
        trunks = json.load(open(out / "crops" / split / "trunk_trees.json"))
        assert set(trunks) == {c["crop"] for c in row["crops"]} and any(trunks.values())
        assert all(set(trunks[f[:-4]]) <= set(np.load(out / "crops" / split / f)["instance_ids"].tolist()) for f in files)
        #whole trees are standing trees that are nearly all inside the window
        whole = json.load(open(out / "crops" / split / "whole_trees.json"))
        assert set(whole) == set(trunks) and all(set(whole[c]) <= set(trunks[c]) for c in trunks)
        crop = np.load(out / "crops" / split / files[0])
        assert len(crop["xyz"]) == len(crop["instance_ids"]) <= cfg["max_crop_points"]
        #most of the flat ground is gone from the crops, and little of the trees
        assert row["removed_as_ground"] > 0.2 and row["tree_points_lost"] < 0.2


def test_download_parts_cover_what_the_notebooks_read():
    assert PARTS["raw"] == ["raw/*"]
    assert {"tiles/*", "crops/*", "summary.json"} == set(PARTS["built"])
