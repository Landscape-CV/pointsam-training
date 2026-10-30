import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")

from treeseg.engine.clicks import BaseSurface, grid_clicks, perfect_clicks, stem_clicks
from treeseg.engine.scoring import click_quality, score_instances, tree_sizes
from treeseg.engine.segment import fuse_masks, segment_tile


def two_trees():
    ids = np.r_[np.full(300, 1), np.full(300, 2), np.full(100, -1)]
    return ids, tree_sizes(ids)


def test_perfect_prediction_scores_one():
    ids, sizes = two_trees()
    pred = np.where(ids > 0, ids - 1, -1)
    r = score_instances(ids, pred, sizes)
    assert r["f1"] == 1.0 and r["tp"] == 2


def test_a_tree_lost_to_ground_removal_is_a_false_negative():
    # the defect found in Oct: dropped trees must still count against recall
    ids, sizes = two_trees()
    work = ids[ids != 2]
    pred = np.where(work > 0, 0, -1)
    r = score_instances(work, pred, sizes)
    assert r["tp"] == 1 and r["fn"] == 1 and r["recall"] == 0.5 and r["precision"] == 1.0


def test_no_predictions_scores_zero():
    ids, sizes = two_trees()
    r = score_instances(ids, np.full(len(ids), -1), sizes)
    assert r["f1"] == 0.0 and r["fn"] == 2


def test_one_mask_over_two_trees_matches_nothing():
    ids, sizes = two_trees()
    r = score_instances(ids, np.where(ids > 0, 0, -1), sizes)  # IoU 0.5 is the edge, so make it lower
    pred = np.zeros(len(ids), dtype=int)
    r = score_instances(ids, pred, sizes)
    assert r["tp"] == 0 and r["fp"] == 1


def test_iou_uses_the_full_tree_size_not_just_kept_points():
    ids = np.full(100, 1)
    r = score_instances(ids, np.zeros(100, dtype=int), {1: 1000})  # 900 points of it were removed
    assert r["tp"] == 0


def test_predictions_on_non_tree_points_can_be_ignored():
    ids, sizes = two_trees()
    pred = np.where(ids > 0, ids - 1, 7)
    assert score_instances(ids, pred, sizes)["fp"] == 1
    r = score_instances(ids, pred, sizes, ignore_non_tree_preds=True)
    assert r["fp"] == 0 and r["preds_ignored"] == 1 and r["f1"] == 1.0


def test_small_fragments_are_not_scored():
    ids = np.r_[np.full(300, 1), np.full(50, 2)]
    assert tree_sizes(ids) == {1: 300}


def test_click_quality_counts_hits_and_misses():
    xyz = np.array([[0.0, 0, 0], [5, 0, 0], [10, 0, 0]])
    ids = np.array([1, 2, -1])
    clicks = [np.array([[0.1, 0, 0]]), np.array([[10.0, 0, 0]]), np.array([[50.0, 0, 0]])]
    r = click_quality(xyz, ids, clicks, {1: 300, 2: 300})
    assert r["clicks"] == 3 and r["trees_hit"] == 1 and r["click_recall"] == 0.5
    assert r["clicks_on_tree"] == pytest.approx(1 / 3)


def test_grid_gives_one_click_per_busy_low_square():
    rng = np.random.default_rng(0)
    xyz = np.c_[rng.uniform(0, 8, 5000), rng.uniform(0, 4, 5000), rng.uniform(0, 10, 5000)]
    clicks = grid_clicks(xyz, spacing=4.0)
    assert len(clicks) >= 2 and all(c.shape == (1, 3) for c in clicks)


def test_perfect_clicks_land_low_on_each_tree():
    rng = np.random.default_rng(0)
    xyz = np.c_[np.r_[np.zeros(300), np.full(300, 5.0)], np.zeros(600), rng.uniform(0, 10, 600)]
    ids = np.r_[np.full(300, 1), np.full(300, 2)]
    clicks = perfect_clicks(xyz, ids)
    assert len(clicks) == 2 and clicks[0][0, 0] == 0.0 and clicks[1][0, 0] == 5.0
    assert all(c[0, 2] < 2 for c in clicks)


def make_trunks():
    rng = np.random.default_rng(0)
    parts = [np.c_[rng.uniform(0, 20, 8000), rng.uniform(0, 20, 8000), rng.uniform(0, 0.1, 8000)]]
    for x, y in [(5, 5), (15, 15)]:
        angle = rng.uniform(0, 2 * np.pi, 3000)
        parts.append(np.c_[x + 0.1 * np.cos(angle), y + 0.1 * np.sin(angle), rng.uniform(0, 8, 3000)])
    return np.vstack(parts)


def test_stems_are_found_at_the_trunks():
    xyz = make_trunks()
    clicks = stem_clicks(xyz)
    centres = sorted((round(c[0, 0]), round(c[0, 1])) for c in clicks)
    assert centres == [(5, 5), (15, 15)]
    assert all(0.3 < c[0, 2] < 2.5 for c in clicks)


def test_base_surface_follows_a_slope():
    x, y = np.meshgrid(np.arange(0, 30, 0.2), np.arange(0, 30, 0.2))
    xyz = np.c_[x.ravel(), y.ravel(), 0.2 * x.ravel()]
    assert np.abs(BaseSurface(xyz).height(xyz)).max() < 0.5


def test_fusion_merges_masks_that_contain_each_others_seed():
    a, b = np.arange(0, 100), np.arange(50, 150)
    labels = fuse_masks(200, [a, b], [np.ones(100), np.ones(100)], [1.0, 1.0], [75, 80], min_points=10)
    assert len(np.unique(labels[labels >= 0])) == 1 and (labels[:150] == labels[0]).all() and (labels[150:] == -1).all()


def test_fusion_keeps_separate_trees_and_gives_shared_points_to_the_higher_logit():
    a, b = np.arange(0, 110), np.arange(100, 210)
    labels = fuse_masks(210, [a, b], [np.ones(110), np.full(110, 2.0)], [1.0, 1.0], [10, 200], min_points=10)
    assert labels[10] != labels[200]
    assert (labels[100:110] == labels[200]).all()


def test_fusion_drops_masks_that_are_too_small():
    labels = fuse_masks(100, [np.arange(5)], [np.ones(5)], [1.0], [0], min_points=50)
    assert (labels == -1).all()


class FakeModel:
    """Stands in for Point-SAM: each click selects the points within 1.5 m of it, in normalised crop terms."""

    def predict_masks(self, coords, features, prompts, labels, prompt_masks=None, multimask_output=True):
        radius = 0.25
        distance = torch.cdist(prompts[:, :1], coords.expand(len(prompts), -1, -1)).squeeze(1)
        logits = radius - distance
        three = torch.stack([logits - 1, logits, logits - 2], dim=1)  # the middle candidate is the good one
        scores = torch.tensor([[0.1, 0.9, 0.2]]).repeat(len(prompts), 1)
        return three, scores


def test_whole_tile_segmentation_on_a_fake_model():
    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([9, 9, 3], 0.3, (2000, 3))])
    clicks = [np.array([[3.0, 3.0, 3.0]]), np.array([[9.0, 9.0, 3.0]])]
    labels = segment_tile(FakeModel(), xyz, clicks, device="cpu")
    assert labels.shape == (4000,)
    assert labels[0] != labels[-1]
    r = score_instances(np.r_[np.full(2000, 1), np.full(2000, 2)], labels, {1: 2000, 2: 2000})
    assert r["f1"] == 1.0
