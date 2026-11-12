"""
dataset.py - serve 12 m crops to Point-SAM.

Each item is one crop reduced to `num_points`, scaled into the unit sphere, with up to `num_masks`
trees as targets and a stem click for each. Augmentation choices follow Kevin Hu's v3 recipe
(Ecomodel jianwei/spcg, pointsam_training/dataset.py). The colour steps are gone: our data has no
colour, so every point gets the same grey feature (0 after Point-SAM's normalisation). Label-based ground
dropping is gone too: crops already had ground removed by the cloth filter, as in real use, so non-tree
plants stay in and the model learns they aren't trees.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def to_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """Centre the points and scale so the furthest one is at distance 1. Point-SAM expects this."""
    centred = xyz - xyz.mean(axis=0)
    radius = float(np.linalg.norm(centred, axis=1).max())
    return (centred / max(radius, 1e-8)).astype(np.float32)


def stem_clicks(xyz: np.ndarray, member: np.ndarray, click_h: float = 1.3, upper_h: float = 4.0) -> tuple[int, int]:
    """Point indices of a click on this tree's stem, and of a second click higher up the same stem."""
    idx = np.flatnonzero(member)
    p = xyz[idx]
    z0 = p[:, 2].min()
    band = idx[(p[:, 2] >= z0 + click_h - 0.5) & (p[:, 2] <= z0 + click_h + 0.7)]
    if len(band) == 0:
        band = idx[p[:, 2] <= np.quantile(p[:, 2], 0.1)]
    axis = np.median(xyz[band, :2], axis=0)

    def nearest(target_h, candidates):
        cost = np.hypot(*(xyz[candidates, :2] - axis).T) + np.abs(xyz[candidates, 2] - (z0 + target_h))
        return int(candidates[np.argmin(cost)])

    upper = idx[(p[:, 2] >= z0 + upper_h - 1.0) & (p[:, 2] <= z0 + upper_h + 1.0)
                & (np.hypot(*(p[:, :2] - axis).T) < 0.8)]
    stem = nearest(click_h, band)
    return stem, (nearest(upper_h, upper) if len(upper) else stem)


def augment(xyz: np.ndarray, ids: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Simulate other scan conditions: lower density, occlusion holes, height-dependent loss."""
    keep = np.ones(len(xyz), dtype=bool)
    if rng.random() < cfg.get("p_density", 0.0):
        keep &= rng.random(len(xyz)) < rng.uniform(0.35, 1.0)
    if rng.random() < cfg.get("p_occlusion", 0.0):
        for _ in range(int(rng.integers(1, 5))):
            centre = xyz[rng.integers(len(xyz))]
            keep &= ~np.all(np.abs(xyz - centre) < rng.uniform(0.3, 1.5, 3), axis=1)
    if rng.random() < cfg.get("p_height", 0.0):
        h = (xyz[:, 2] - xyz[:, 2].min()) / max(1e-6, np.ptp(xyz[:, 2]))
        k = rng.uniform(0.3, 0.8)
        #ground scanners lose the canopy top, airborne ones lose the stems
        keep &= rng.random(len(xyz)) < (1 - k * h if rng.random() < 0.6 else 1 - k * (1 - h))
    if keep.sum() >= 256:
        xyz, ids = xyz[keep], ids[keep]

    jitter = cfg.get("jitter", 0.0)
    if jitter > 0:
        xyz = xyz + rng.normal(0, rng.uniform(0, jitter), xyz.shape)
    tilt = np.deg2rad(cfg.get("tilt_deg", 0.0))
    if tilt > 0:
        ax, ay = rng.uniform(-tilt, tilt, 2)
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        xyz = xyz @ (rx @ ry).T
    angle = rng.uniform(-np.pi, np.pi)
    rz = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return (xyz @ rz.T) * rng.uniform(0.85, 1.05), ids


def subsample(xyz: np.ndarray, ids: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exactly n points. At least half come from trees when a crop has enough, so trees stay well sampled."""
    if len(xyz) <= n:
        idx = np.r_[np.arange(len(xyz)), rng.choice(len(xyz), n - len(xyz), replace=True)]
        return xyz[idx], ids[idx]
    trees = np.flatnonzero(ids > 0)
    picked = rng.choice(trees, min(len(trees), n // 2), replace=False)
    rest = np.setdiff1d(np.arange(len(xyz)), picked, assume_unique=False)
    idx = np.r_[picked, rng.choice(rest, n - len(picked), replace=False)]
    return xyz[idx], ids[idx]


def choose_masks(ids: np.ndarray, num_masks: int, min_points: int, rng: np.random.Generator, allowed=None) -> np.ndarray:
    """Boolean masks (num_masks, N), one tree each. Trees repeat if the crop has fewer than num_masks.
    With allowed (tree ids), only those trees are targets, unless the crop has none of them."""
    trees, counts = np.unique(ids[ids > 0], return_counts=True)
    #a tiny fragment at the crop edge is not a fair target, unless it is all the crop has
    big = trees[counts >= min_points]
    trees = big if len(big) else trees
    if allowed is not None:
        standing = trees[np.isin(trees, allowed)]
        trees = standing if len(standing) else trees
    if len(trees) == 0:
        masks = np.zeros((num_masks, len(ids)), dtype=bool)
    else:
        chosen = rng.choice(trees, num_masks, replace=len(trees) < num_masks)
        masks = np.stack([ids == t for t in chosen])
    #Point-SAM's click sampler needs every mask to have both inside and outside points
    for mask in masks:
        if mask.all() or not mask.any():
            flip = rng.choice(len(ids), min(32, len(ids)), replace=False)
            mask[flip] = not mask.all()
    return masks


class ForestCrops(Dataset):
    def __init__(self, crop_dir: str, num_points: int = 10000, num_masks: int = 8, min_mask_points: int = 40,
                 augmentation: dict | None = None, resample_each_epoch: bool = False, seed: int = 0, targets: str = "all"):
        """targets "all" lets any tree in the crop be a target, cut pieces included. "trunk_in_crop" keeps only trees
        that stand inside the crop's window, read from trunk_trees.json, which the dataset build writes."""
        self.files = sorted(glob.glob(os.path.join(crop_dir, "*.npz")))
        if not self.files:
            raise FileNotFoundError(f"no crops in {crop_dir}")
        if targets not in ("all", "trunk_in_crop"):
            raise ValueError(f"targets must be all or trunk_in_crop, got {targets!r}")
        self.trunks = None
        if targets == "trunk_in_crop":
            path = os.path.join(crop_dir, "trunk_trees.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} is missing: download the dataset again, or rebuild it with 00_dataset.ipynb")
            with open(path) as f:
                self.trunks = json.load(f)
        self.num_points = num_points
        self.num_masks = num_masks
        self.min_mask_points = min_mask_points
        self.augmentation = augmentation
        self.resample_each_epoch = resample_each_epoch  # new sampled points and target trees each epoch
        self.seed = seed
        self.epoch = 0  # the trainer moves this on, so training randomness changes every epoch

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        with np.load(self.files[index]) as crop:
            xyz, ids = crop["xyz"].astype(np.float64), crop["instance_ids"]
        rng = np.random.default_rng(self.seed + index * 10007 + (self.epoch * 7919 if self.resample_each_epoch else 0))
        if self.augmentation:
            xyz, ids = augment(xyz, ids, self.augmentation, rng)
        xyz, ids = subsample(xyz, ids, self.num_points, rng)
        allowed = None if self.trunks is None else self.trunks[os.path.splitext(os.path.basename(self.files[index]))[0]]
        masks = choose_masks(ids, self.num_masks, self.min_mask_points, rng, allowed)
        stems = np.array([stem_clicks(xyz, mask) for mask in masks], dtype=np.int64)
        return {"coords": torch.from_numpy(to_unit_sphere(xyz)),
                "features": torch.zeros(self.num_points, 3),
                "gt_masks": torch.from_numpy(masks),
                "stem_idx": torch.from_numpy(stems)}
