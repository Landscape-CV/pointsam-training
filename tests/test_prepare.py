import sys

import numpy as np

sys.path.insert(0, "src")

from treeseg.data.prepare import ground_by_tile, tile_grid, voxelize, window_starts, write_crops


def test_voxelize_keeps_one_point_per_voxel():
    xyz = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.5, 0, 0]])
    out_xyz, _ = voxelize(xyz, np.array([1, 1, 2]), size=0.05)
    assert len(out_xyz) == 2


def test_voxelize_takes_the_most_common_label():
    xyz = np.array([[0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
    out_xyz, out_ids = voxelize(xyz, np.array([7, 9, 9]), size=0.05)
    assert out_ids.tolist() == [9]
    # the kept point must actually carry that label, not just be in the voxel
    assert out_xyz[0, 0] in (0.02, 0.03)


def test_voxelize_keeps_points_and_labels_together():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 5, (5000, 3))
    ids = (xyz[:, 0] > 2.5).astype(np.int32) + 1  # label decided by position
    out_xyz, out_ids = voxelize(xyz, ids, size=0.05)
    assert np.array_equal(out_ids, (out_xyz[:, 0] > 2.5).astype(np.int32) + 1)


def test_windows_reach_the_far_edge_without_slivers():
    starts = window_starts(0.0, 27.1, size=12.0, stride=6.0)
    assert starts[0] == 0.0
    assert abs(starts[-1] + 12.0 - 27.1) < 1e-9
    assert np.diff(starts).max() <= 6.0


def test_short_axis_gets_a_single_window():
    assert window_starts(0.0, 8.0, size=12.0, stride=6.0) == [0.0]


def test_crops_stay_inside_their_window_with_labels_aligned(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 30, (20000, 3))
    ids = np.floor(xyz[:, 0] / 5).astype(np.int32) + 1  # trees are x bands, so labels can be checked
    records = write_crops(xyz, ids, str(tmp_path), "t", 12.0, 6.0, min_points=100, max_points=500, min_trees=1)
    assert records
    for r in records:
        z = np.load(tmp_path / f"{r['crop']}.npz")
        assert len(z["xyz"]) == len(z["instance_ids"]) == r["n_points"] <= 500
        assert z["xyz"][:, 0].min() >= r["origin_xy"][0] - 1e-3
        assert z["xyz"][:, 0].max() < r["origin_xy"][0] + 12.0 + 1e-3
        assert np.array_equal(z["instance_ids"], np.floor(z["xyz"][:, 0] / 5).astype(np.int32) + 1)


def test_crops_with_too_few_trees_are_skipped(tmp_path):
    xyz = np.random.default_rng(0).uniform(0, 12, (5000, 3))
    ids = np.full(5000, -1, dtype=np.int32)  # no trees at all
    assert write_crops(xyz, ids, str(tmp_path), "t", 12.0, 6.0, 100, 500, min_trees=1) == []


def test_tile_grid_puts_the_far_edge_in_the_last_tile():
    xyz = np.array([[0.0, 0, 0], [29.9, 0, 0], [30.0, 0, 0], [60.0, 0, 0]])
    tiles = tile_grid(xyz, 30.0)
    assert tiles[0] == tiles[1] and tiles[1] != tiles[2] and tiles[2] == tiles[3]


def test_ground_by_tile_removes_a_flat_floor_but_keeps_what_stands_on_it():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 40, (20000, 2)), rng.normal(0, 0.02, 20000)]
    trunk = np.c_[rng.normal(20, 0.1, (3000, 2)), rng.uniform(2, 10, 3000)]
    xyz = np.vstack([floor, trunk])
    ground = ground_by_tile(xyz, 30.0, {"resolution": 0.5, "threshold": 0.5, "iterations": 500, "rigidness": 1})
    assert ground[:20000].mean() > 0.95
    assert ground[20000:].mean() < 0.02
