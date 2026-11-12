import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

sys.path.insert(0, "src")

from treeseg.data.dataset import ForestCrops, choose_masks, stem_clicks, subsample, to_unit_sphere
from treeseg.engine.trainer import first_clicks, lr_factor
from treeseg.models.lora import LinearWithLoRA, add_lora, merged_lora
from treeseg.utils.curves import make_all, per_epoch


def make_crop(path, n=3000, trees=4):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 12, (n, 3)).astype(np.float32)
    xyz[:, 2] *= 2
    ids = (np.floor(xyz[:, 0] / (12 / trees)).astype(np.int32) + 1)
    ids[xyz[:, 2] < 1] = -1  # low points are not tree
    np.savez(path, xyz=xyz, instance_ids=ids)


def test_lora_is_a_no_op_at_start():
    base = nn.Linear(16, 16)
    x = torch.randn(2, 5, 16)
    assert torch.equal(LinearWithLoRA(base, rank=4)(x), base(x))


def test_lora_freezes_base_and_trains_adapters():
    wrapped = LinearWithLoRA(nn.Linear(16, 16), rank=4)
    assert not wrapped.base.weight.requires_grad
    assert wrapped.lora_a.requires_grad and wrapped.lora_b.requires_grad


def test_merged_layer_gives_the_same_output():
    wrapped = LinearWithLoRA(nn.Linear(16, 16), rank=4, alpha=8)
    nn.init.normal_(wrapped.lora_b)
    x = torch.randn(3, 16)
    assert torch.allclose(wrapped(x), wrapped.merged()(x), atol=1e-5)


def test_saved_keys_match_stock_and_adapters_return():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.other = nn.Linear(4, 4), nn.Linear(4, 4)

    model = nn.Sequential(Block(), Block())
    stock_keys = set(model.state_dict())
    assert add_lora(model, targets=("q_proj",), rank=2) == 2
    with merged_lora(model):
        assert set(model.state_dict()) == stock_keys
    assert isinstance(model[0].q_proj, LinearWithLoRA)
    assert isinstance(model[0].other, nn.Linear)


def test_unit_sphere_furthest_point_is_at_one():
    out = to_unit_sphere(np.random.default_rng(0).uniform(-5, 9, (100, 3)))
    assert np.linalg.norm(out, axis=1).max() == pytest.approx(1.0, abs=1e-5)
    assert np.abs(out.mean(axis=0)).max() < 1e-5


def test_subsample_gives_exact_count_and_keeps_labels_aligned():
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 1, (500, 3))
    ids = np.where(xyz[:, 0] > 0.5, 1, -1)
    for n in (100, 800):
        out_xyz, out_ids = subsample(xyz, ids, n, rng)
        assert len(out_xyz) == n
        assert np.array_equal(out_ids, np.where(out_xyz[:, 0] > 0.5, 1, -1))


def test_masks_have_inside_and_outside_points():
    rng = np.random.default_rng(0)
    for ids in (np.ones(200, dtype=int), -np.ones(200, dtype=int), np.r_[np.ones(100, dtype=int), 2 * np.ones(100, dtype=int)]):
        masks = choose_masks(ids, 8, 40, rng)
        assert masks.shape == (8, 200)
        assert all(m.any() and not m.all() for m in masks)


def test_tiny_fragments_are_not_targets_when_real_trees_exist():
    ids = np.r_[np.full(500, 1), np.full(5, 2), np.full(50, -1)]
    masks = choose_masks(ids, 4, 40, np.random.default_rng(0))
    assert all(m[:500].all() and not m[500:505].any() for m in masks)


def test_stem_click_is_low_on_its_own_tree():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 15, 2000)
    xyz = np.c_[rng.normal(0, 0.1, 2000), rng.normal(0, 0.1, 2000), z]
    member = np.ones(2000, dtype=bool)
    member[:100] = False
    stem, upper = stem_clicks(xyz, member)
    assert member[stem] and member[upper]
    assert 0.5 < xyz[stem, 2] < 2.5 and 2.5 < xyz[upper, 2] < 5.5


def test_dataset_item_has_the_shapes_point_sam_expects(tmp_path):
    make_crop(tmp_path / "a.npz")
    item = ForestCrops(str(tmp_path), num_points=1000, num_masks=8, augmentation={"p_density": 0.5, "jitter": 0.01, "tilt_deg": 3})[0]
    assert item["coords"].shape == (1000, 3)
    assert item["features"].shape == (1000, 3) and not item["features"].any()
    assert item["gt_masks"].shape == (8, 1000) and item["gt_masks"].dtype == torch.bool
    assert item["stem_idx"].shape == (8, 2)
    for mask, stem in zip(item["gt_masks"], item["stem_idx"]):
        assert mask[stem[0]] and mask[stem[1]]


def test_training_randomness_changes_by_epoch_but_validation_does_not(tmp_path):
    make_crop(tmp_path / "a.npz")
    train = ForestCrops(str(tmp_path), num_points=500, augmentation={"jitter": 0.01}, resample_each_epoch=True)
    first = train[0]["coords"].clone()
    train.epoch = 1
    assert not torch.equal(first, train[0]["coords"])
    val = ForestCrops(str(tmp_path), num_points=500)
    first = val[0]["coords"].clone()
    val.epoch = 1
    assert torch.equal(first, val[0]["coords"])


def test_with_augmentation_off_points_are_unchanged_but_still_resampled(tmp_path):
    make_crop(tmp_path / "a.npz")
    train = ForestCrops(str(tmp_path), num_points=500, augmentation=None, resample_each_epoch=True)
    first = train[0]["coords"].clone()
    train.epoch = 1
    second = train[0]["coords"]
    assert not torch.equal(first, second)  # a different sample of points
    assert first.shape == second.shape


def test_first_clicks_land_inside_the_mask():
    coords = torch.rand(1, 100, 3)
    gt = torch.zeros(1, 2, 100, dtype=torch.bool)
    gt[0, 0, :50], gt[0, 1, 50:] = True, True
    stem_idx = torch.tensor([[[3, 7], [60, 70]]])
    points, labels = first_clicks(coords, gt, stem_idx, "stem")
    assert points.shape == (2, 2, 3) and labels.all()
    assert torch.equal(points[0, 0], coords[0, 3]) and torch.equal(points[1, 1], coords[0, 70])
    points, _ = first_clicks(coords, gt, stem_idx, "random")
    for mask_index in range(2):
        hit = (coords[0] == points[mask_index, 0]).all(dim=1).nonzero().item()
        assert gt[0, mask_index, hit]
    assert first_clicks(coords, gt, stem_idx, "interior") == (None, None)


def test_learning_rate_warms_up_then_decays_to_the_floor():
    assert lr_factor(0, 3000, 150, 0.05) == pytest.approx(1 / 150)
    assert lr_factor(149, 3000, 150, 0.05) == pytest.approx(1.0)
    assert lr_factor(150, 3000, 150, 0.05) == pytest.approx(1.0)
    assert lr_factor(3000, 3000, 150, 0.05) == pytest.approx(0.05)


def fake_history(steps=(25, 50, 75, 100, 125, 150, 175, 200), crops=100):
    rows = [{"step": s, "epoch": s * 2 / crops, "split": "train", "loss": 1 / s ** 0.2, "iou_first": s / 300} for s in steps]
    rows += [{"step": s, "epoch": s * 2 / crops, "split": "val", "val_loss": 0.9 - s / 500, "val_iou": 0.5, "val_iou_first": 0.4,
              "val_iou_pick": s / 500} for s in steps[3::4]]
    return pd.DataFrame(rows).sort_values("step")


def test_per_epoch_table_counts_whole_epochs():
    table = per_epoch(fake_history())
    assert table["whole_epoch"].tolist() == [1, 2, 3, 4]
    assert table["train_loss"].notna().all()
    assert table.loc[table["whole_epoch"] == 4, "val_iou_pick"].item() == pytest.approx(0.4)


def test_curves_are_written_from_a_partial_run(tmp_path):
    history = tmp_path / "history.csv"
    fake_history().to_csv(history, index=False)
    paths = make_all(str(history), str(tmp_path / "figures"))
    for path in paths.values():
        assert (tmp_path / "figures" / path.split("/")[-1]).stat().st_size > 0


def test_only_trees_standing_in_the_crop_are_targets_when_asked(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.uniform(0, 12, (3000, 3)).astype(np.float32)
    ids = np.r_[np.full(1000, 1), np.full(1000, 2), np.full(1000, -1)].astype(np.int32)
    np.savez(tmp_path / "a_crop000.npz", xyz=xyz, instance_ids=ids)
    with pytest.raises(FileNotFoundError):
        ForestCrops(str(tmp_path), num_points=1000, targets="trunk_in_crop")
    (tmp_path / "trunk_trees.json").write_text('{"a_crop000": [2]}')
    item = ForestCrops(str(tmp_path), num_points=1000, num_masks=8, targets="trunk_in_crop")[0]
    assert len(np.unique(item["gt_masks"].numpy(), axis=0)) == 1  # every target is tree 2
    everything = ForestCrops(str(tmp_path), num_points=1000, num_masks=8)[0]
    assert len(np.unique(everything["gt_masks"].numpy(), axis=0)) == 2
    #a crop with no standing tree keeps its usual targets, so no crop is lost
    (tmp_path / "trunk_trees.json").write_text('{"a_crop000": []}')
    assert ForestCrops(str(tmp_path), num_points=1000, num_masks=8, targets="trunk_in_crop")[0]["gt_masks"].any()
