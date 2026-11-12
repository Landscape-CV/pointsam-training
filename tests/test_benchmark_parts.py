import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")

from treeseg.engine.clicks import BaseSurface, grid_clicks, perfect_clicks, stem_clicks
from treeseg.engine.scoring import click_quality, match_instances, score_instances, score_raw_masks, tree_sizes
from treeseg.engine.segment import click_candidates, fuse_masks, load_masks, save_masks, segment_tile


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


def test_mean_tree_iou_counts_missed_trees_but_matched_iou_does_not():
    # 10 trees, and only one found perfectly
    ids = np.repeat(np.arange(1, 11), 300)
    pred = np.where(ids == 1, 0, -1)
    r = score_instances(ids, pred, tree_sizes(ids))
    assert r["mean_matched_iou"] == 1.0
    assert r["mean_tree_iou"] == pytest.approx(0.1)


def test_mean_tree_iou_is_zero_with_no_predictions():
    ids, sizes = two_trees()
    assert score_instances(ids, np.full(len(ids), -1), sizes)["mean_tree_iou"] == 0.0


def test_click_demo_returns_three_masks_and_picks_the_highest_score():
    from treeseg.engine.click_demo import click_positions, predict_one_click

    rng = np.random.default_rng(0)
    coords = rng.uniform(-0.5, 0.5, (2000, 3)).astype(np.float32)
    truth = np.linalg.norm(coords - np.array([0.1, 0.1, 0.0]), axis=1) < 0.25
    masks = np.stack([truth, ~truth])
    stems = np.array([[int(np.flatnonzero(truth)[0])] * 2, [int(np.flatnonzero(~truth)[0])] * 2])
    item = {"coords": torch.from_numpy(coords), "features": torch.zeros(2000, 3), "gt_masks": torch.from_numpy(masks), "stem_idx": torch.from_numpy(stems)}

    class Model(FakeModel):
        def eval(self):
            return self

    result = predict_one_click(Model(), item, 0, "stem1", device="cpu")
    assert result["masks"].shape == (3, 2000) and result["picked"] == 1
    assert len(result["ious"]) == 3 and 0 <= result["ious"].min() <= result["ious"].max() <= 1
    positions = click_positions(item, 0)
    assert set(positions) == {"stem", "upper", "random"}


def test_click_plots_draw_without_error():
    import matplotlib
    matplotlib.use("Agg")
    from treeseg.utils.view3d import plot_click_kinds, plot_mask_candidates

    rng = np.random.default_rng(0)
    coords = rng.uniform(-0.5, 0.5, (500, 3)).astype(np.float32)
    truth = coords[:, 0] > 0
    item = {"coords": torch.from_numpy(coords), "gt_masks": torch.from_numpy(truth[None])}
    plot_click_kinds(item, {"stem": coords[0], "upper": coords[1], "random": coords[2]})
    result = {"coords": coords, "truth": truth, "masks": np.stack([truth, ~truth, truth]), "ious": np.array([1.0, 0.0, 1.0]),
              "scores": np.array([0.5, 0.9, 0.1]), "picked": 1, "click": coords[:1]}
    plot_mask_candidates(result)


def test_clicks_land_on_the_trunk_near_the_requested_height():
    from treeseg.engine.clicks import clicks_at_positions

    xyz = make_trunks()
    clicks = clicks_at_positions(xyz, np.array([[5.0, 5.0], [15.0, 15.0], [99.0, 99.0]]))
    assert len(clicks) == 2  # nothing is near the third position, so it makes no click
    assert all(0.8 < c[0, 2] < 1.8 for c in clicks)
    assert np.hypot(clicks[0][0, 0] - 5, clicks[0][0, 1] - 5) < 0.3


def test_every_click_method_is_selected_by_name():
    from treeseg.engine.clicks import CLICK_METHODS, make_clicks

    assert {"grid", "stems", "treex", "perfect"} <= set(CLICK_METHODS)
    with pytest.raises(ValueError):
        make_clicks("nope", np.zeros((1, 3)), np.zeros(1))


def test_matching_lists_the_pairs_the_score_counts():
    ids, sizes = two_trees()
    pred = np.r_[np.full(300, 7), np.full(150, 9), np.full(250, -1)]  # tree 1 found, tree 2 half found
    match = match_instances(ids, pred, sizes)
    assert match["matches"][1] == (7, 1.0) and match["matches"][2] == (9, 0.5)
    assert score_instances(ids, pred, sizes)["tp"] == len(match["matches"]) == 2
    assert 2 not in match_instances(ids, pred, sizes, iou_threshold=0.6)["matches"]


def test_matching_reports_ignored_predictions():
    ids, sizes = two_trees()
    pred = np.r_[np.full(600, -1), np.full(100, 3)]  # only on non-tree points
    assert match_instances(ids, pred, sizes, ignore_non_tree_preds=True)["ignored"] == [3]


def test_raw_masks_are_scored_one_tree_at_a_time():
    ids, sizes = two_trees()
    raw = score_raw_masks(ids, [np.arange(0, 300), np.arange(300, 400), np.arange(600, 700)], sizes)
    assert raw["best"][1] == (0, 1.0) and raw["best"][2][0] == 1 and raw["best"][2][1] == pytest.approx(1 / 3)
    assert raw["raw_trees_found"] == 1 and raw["raw_recall"] == 0.5 and raw["raw_masks"] == 3
    assert score_raw_masks(ids, [], sizes)["raw_recall"] == 0.0


def test_fusion_reports_which_masks_were_joined_and_dropped():
    a, b, far, tiny = np.arange(0, 100), np.arange(50, 150), np.arange(300, 400), np.arange(500, 505)
    labels, groups = fuse_masks(600, [a, b, far, tiny], [np.ones(len(m)) for m in (a, b, far, tiny)], [1.0] * 4, [75, 80, 350, 502],
                                min_points=10, return_groups=True)
    assert groups[0] == groups[1] == labels[0] and groups[2] == labels[350] != groups[0] and groups[3] == -1
    assert fuse_masks(10, [], [], [], [], return_groups=True)[1].shape == (0,)


def test_saved_masks_merge_to_the_same_labels(tmp_path):
    clicks = [np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 5.0]]), np.array([[9.0, 9.0, 1.0]])]
    masks, logits = [np.arange(0, 100), np.arange(200, 320)], [np.linspace(0.5, 2, 100), np.linspace(1, 3, 120)]
    keep = np.arange(1000) % 3 > 0
    save_masks(str(tmp_path / "m.npz"), clicks, masks, logits, [0.5, 0.75], [10, 250], keep=keep)
    saved = load_masks(str(tmp_path / "m.npz"))
    assert (saved["keep"] == keep).all()
    assert len(saved["clicks"]) == 2 and saved["clicks"][0].shape == (2, 3) and np.allclose(saved["clicks"][1], clicks[1])
    again = fuse_masks(400, saved["masks"], saved["logits"], saved["scores"], saved["seeds"], min_points=10)
    assert (again == fuse_masks(400, masks, logits, [0.5, 0.75], [10, 250], min_points=10)).all()
    save_masks(str(tmp_path / "none.npz"), [], [], [], [], [])
    assert load_masks(str(tmp_path / "none.npz"))["masks"] == [] and load_masks(str(tmp_path / "none.npz"))["keep"] is None


def test_one_click_gives_three_candidates_on_the_sampled_points():
    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([30, 30, 3], 0.3, (500, 3))])
    result = click_candidates(FakeModel(), xyz, np.array([[3.0, 3.0, 3.0]]), crop_points=1000, device="cpu")
    assert len(result["crop_index"]) == 2000 and result["masks"].shape == (3, 1000) and result["picked"] == 1


def test_benchmark_scores_baselines_and_clicks_before_and_after_merging(tmp_path, monkeypatch):
    from treeseg.engine import benchmark

    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([9, 9, 3], 0.3, (2000, 3))])
    ids = np.r_[np.full(2000, 1), np.full(2000, 2)]
    tile = {"name": "t", "xyz": xyz, "ids": ids, "sizes": {1: 2000, 2: 2000}, "full_xyz": xyz, "keep": np.ones(4000, dtype=bool)}
    monkeypatch.setattr(benchmark, "prepare_tile", lambda *args: tile)
    monkeypatch.setattr(benchmark, "build_point_sam", lambda *args: FakeModel())
    monkeypatch.setattr(FakeModel, "to", lambda self, device: self, raising=False)
    monkeypatch.setattr(FakeModel, "eval", lambda self: self, raising=False)
    #a baseline that finds only the first tree
    monkeypatch.setitem(benchmark.BASELINES, "half", lambda tile: (np.where(tile["ids"] == 1, 0, -1), [np.array([[3.0, 3.0, 3.0]])]))
    cfg = {"out_dir": str(tmp_path), "tile_dir": "", "tiles": ["t"], "csf": {}, "min_tree_points": 200, "checkpoints": {"fake": ""},
           "click_methods": ["perfect"], "baselines": ["half"], "save_masks": True, "iou_threshold": 0.5, "crop_size": 12.0,
           "crop_stride": 4.0, "crop_points": 10000, "group_number": 512, "group_size": 64}
    table = benchmark.summarise(benchmark.run_benchmark(cfg, device="cpu"))
    strict = table[table["rule"] == "strict"].set_index("checkpoint")
    assert strict.loc["half", "recall"] == 0.5 and np.isnan(strict.loc["half", "raw_recall"])
    assert strict.loc["fake", "f1"] == 1.0 and strict.loc["fake", "raw_recall"] == 1.0
    saved = load_masks(str(tmp_path / "fake_t_perfect_masks.npz"))
    assert len(saved["masks"]) == len(saved["clicks"]) == 2 and (tmp_path / "half_t_none_pred.npy").exists()


class RecordingModel(FakeModel):
    """Keeps the prompts it was given, so a test can see which clicks were sent as positive and negative."""

    def predict_masks(self, coords, features, prompts, labels, prompt_masks=None, multimask_output=True):
        self.prompts, self.labels = prompts, labels
        return super().predict_masks(coords, features, prompts, labels, prompt_masks, multimask_output)


def test_neighbouring_clicks_are_sent_as_negative_clicks():
    from treeseg.engine.segment import predict_click_masks

    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([5, 3, 3], 0.3, (2000, 3)), rng.normal([30, 30, 3], 0.3, (2000, 3))])
    clicks = [np.array([[3.0, 3.0, 3.0]]), np.array([[5.0, 3.0, 3.0]]), np.array([[30.0, 30.0, 3.0]])]
    model = RecordingModel()
    predict_click_masks(model, xyz[:4000], clicks[:2], device="cpu", crop_stride=12.0)
    assert model.labels.tolist() == [[1], [1]]  # no negatives unless asked for
    predict_click_masks(model, xyz[:4000], clicks[:2], device="cpu", crop_stride=12.0, negative_clicks=3)
    #each of the two close trees gets its own click as positive and the other's as negative
    assert model.labels.tolist() == [[1, 0], [1, 0]]
    assert torch.allclose(model.prompts[0, 1], model.prompts[1, 0]) and torch.allclose(model.prompts[0, 0], model.prompts[1, 1])
    #a tree further away than the radius is never a negative
    predict_click_masks(model, xyz, clicks, device="cpu", crop_stride=12.0, negative_clicks=3, negative_radius=6.0)
    assert model.labels.sum(dim=1).min() >= 1 and model.labels.shape[1] <= 2


def test_a_crop_can_be_centred_on_each_click():
    from treeseg.engine.segment import predict_click_masks

    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([5, 3, 3], 0.3, (2000, 3))])
    clicks = [np.array([[3.0, 3.0, 3.0]]), np.array([[5.0, 3.0, 3.0]])]
    model = RecordingModel()
    predict_click_masks(model, xyz, clicks, device="cpu", crop_stride=12.0)
    assert model.prompts.shape[0] == 2  # on the grid, both clicks share one crop
    masks, _, _, seeds = predict_click_masks(model, xyz, clicks, device="cpu", crop_stride=12.0, centre_on_click=True)
    assert model.prompts.shape[0] == 1 and len(masks) == len(seeds) == 2  # one crop per click


class WrongChoiceModel(FakeModel):
    """Rates its tight first candidate highest, though the second one is the whole tree."""

    def predict_masks(self, coords, features, prompts, labels, prompt_masks=None, multimask_output=True):
        three, _ = super().predict_masks(coords, features, prompts, labels, prompt_masks, multimask_output)
        three = torch.stack([three[:, 1] - 0.1, three[:, 1], three[:, 2]], dim=1)
        return three, torch.tensor([[0.9, 0.1, 0.2]]).repeat(len(prompts), 1)


def test_the_kept_candidate_can_be_the_models_choice_the_first_or_the_best_given_the_truth():
    from treeseg.engine.segment import predict_click_masks

    rng = np.random.default_rng(0)
    xyz = rng.normal([3, 3, 3], 0.3, (2000, 3))
    click = [np.array([[3.0, 3.0, 3.0]])]
    whole = predict_click_masks(FakeModel(), xyz, click, device="cpu")[0][0]
    ids = np.full(len(xyz), -1)
    ids[whole] = 7
    by_score = predict_click_masks(WrongChoiceModel(), xyz, click, device="cpu")[0][0]
    by_first = predict_click_masks(WrongChoiceModel(), xyz, click, device="cpu", pick="first")[0][0]
    by_truth = predict_click_masks(WrongChoiceModel(), xyz, click, device="cpu", pick="truth", truth_ids=ids)[0][0]
    assert 0 < len(by_score) < len(whole) and np.array_equal(by_first, by_score)
    assert np.array_equal(np.sort(by_truth), np.sort(whole))
    with pytest.raises(ValueError):
        predict_click_masks(FakeModel(), xyz, click, device="cpu", pick="truth")


def test_variants_get_their_own_rows_and_ground_removal_is_saved_once(tmp_path, monkeypatch):
    from treeseg.engine import benchmark

    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.normal([3, 3, 3], 0.3, (2000, 3)), rng.normal([9, 9, 3], 0.3, (2000, 3))])
    ids = np.r_[np.full(2000, 1), np.full(2000, 2)]
    calls = []
    monkeypatch.setattr(benchmark, "load_tile", lambda *args: (xyz, ids))
    monkeypatch.setattr(benchmark, "csf_ground_mask", lambda points, **csf: calls.append(1) or np.zeros(len(points), dtype=bool))
    monkeypatch.setattr(benchmark, "build_point_sam", lambda *args: FakeModel())
    monkeypatch.setattr(FakeModel, "to", lambda self, device: self, raising=False)
    monkeypatch.setattr(FakeModel, "eval", lambda self: self, raising=False)
    cfg = {"out_dir": str(tmp_path / "out"), "ground_cache": str(tmp_path / "ground"), "tile_dir": "", "tiles": ["t"], "csf": {"resolution": 0.5},
           "min_tree_points": 200, "checkpoints": {"fake": ""}, "click_methods": ["perfect"], "iou_threshold": 0.5, "crop_size": 12.0,
           "crop_stride": 4.0, "crop_points": 10000, "group_number": 512, "group_size": 64,
           "variants": [{"name": "base"}, {"name": "centred", "centre_on_click": True}, {"name": "best", "pick": "truth"}]}
    table = benchmark.summarise(benchmark.run_benchmark(cfg, device="cpu"))
    assert sorted(table["checkpoint"].unique()) == ["fake_base", "fake_best", "fake_centred"]
    benchmark.run_benchmark(cfg, device="cpu")
    assert len(calls) == 1 and (tmp_path / "ground" / "t_keep_resolution0.5.npy").exists()  # the second run reused the saved one
