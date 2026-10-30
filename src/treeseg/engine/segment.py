"""
segment.py - turn clicks into tree labels for a whole tile.

Ported from Kevin Hu's SegmenterPointSAM (local mode) and fuse_stem_masks. Each click runs in a 12 m crop
around it, so the model sees what it trained on. The per-click masks are then merged into one label per point.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def predict_click_masks(model, xyz: np.ndarray, click_groups: list[np.ndarray], crop_size: float = 12.0, crop_stride: float = 4.0,
                        crop_points: int = 10000, device: str = "cuda", seed: int = 0) -> tuple[list, list, list, list]:
    """Run the model. Returns, per click group: mask point indices, their logits, a score, and the seed point."""
    rng = np.random.default_rng(seed)
    point_tree = cKDTree(xyz)
    xy_tree = cKDTree(xyz[:, :2])
    origin = xyz[:, :2].min(axis=0)
    half = crop_size / 2

    #clicks sharing a stride cell share one crop, which keeps a click within stride/2 of its crop centre
    crops: dict[tuple, list[int]] = {}
    for g, group in enumerate(click_groups):
        crops.setdefault(tuple(np.floor((group[0, :2] - origin) / crop_stride).astype(int).tolist()), []).append(g)

    masks, logits, scores, seeds = [], [], [], []
    for cell, group_ids in crops.items():
        centre = origin + (np.asarray(cell) + 0.5) * crop_stride
        idx = np.asarray(xy_tree.query_ball_point(centre, r=half * np.sqrt(2)), dtype=np.int64)
        if len(idx) == 0:
            continue
        idx = idx[(np.abs(xyz[idx, :2] - centre) <= half).all(axis=1)]
        if len(idx) < 16:
            continue
        sample = idx if len(idx) <= crop_points else rng.choice(idx, crop_points, replace=False)
        points = xyz[sample]
        mean = points.mean(axis=0)
        radius = float(np.linalg.norm(points - mean, axis=1).max()) or 1.0
        coords = torch.from_numpy(((points - mean) / radius).astype(np.float32)).unsqueeze(0).to(device)
        features = torch.zeros_like(coords)  # no colour: the same grey as in training

        #different click counts are padded by repeating a group's first click
        most = max(len(click_groups[g]) for g in group_ids)
        padded = np.stack([np.vstack([click_groups[g], np.repeat(click_groups[g][:1], most - len(click_groups[g]), 0)]) for g in group_ids])
        prompts = np.clip((padded - mean) / radius, -1.0, 1.0).astype(np.float32)
        prompt_labels = np.ones(prompts.shape[:2], dtype=np.int64)
        with torch.no_grad():
            out_masks, out_scores = model.predict_masks(coords, features, torch.from_numpy(prompts).to(device),
                                                        torch.from_numpy(prompt_labels).to(device), prompt_masks=None, multimask_output=True)
        #of the 3 candidates keep the one the model rates highest, as inference has no answer to check against
        best = torch.argmax(out_scores, dim=1)
        row = torch.arange(len(group_ids), device=out_scores.device)
        crop_logits = out_masks[row, best].float().cpu().numpy()
        crop_scores = out_scores[row, best].float().cpu().numpy()

        #the sample has logits, so every other crop point takes its nearest sampled point's logit
        if len(sample) < len(idx):
            crop_logits = crop_logits[:, cKDTree(points).query(xyz[idx])[1]]
        for k, g in enumerate(group_ids):
            on = crop_logits[k] > 0
            masks.append(idx[on])
            logits.append(crop_logits[k][on])
            scores.append(float(crop_scores[k]))
            seeds.append(int(point_tree.query(click_groups[g][0])[1]))
    return masks, logits, scores, seeds


def fuse_masks(n_points: int, masks: list, logits: list, scores: list, seeds: list, containment: float = 0.8, min_points: int = 50) -> np.ndarray:
    """One label per point. Masks are the same tree if each contains the other's seed, or one sits at least
    `containment` inside the other. A point claimed by several trees goes to the highest logit."""
    k = len(masks)
    if k == 0:
        return np.full(n_points, -1, dtype=np.int32)
    parent = list(range(k))

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    #each mask as a sorted set, so overlap and seed checks are fast
    sets, confidences = [], []
    for mask, logit, score in zip(masks, logits, scores):
        points, first = np.unique(np.asarray(mask, dtype=np.int64), return_index=True)
        sets.append(points)
        confidences.append(np.asarray(logit, dtype=np.float32)[first] + 1e-3 * score)
    sizes = np.array([len(s) for s in sets])
    owner_point = np.concatenate(sets) if sizes.sum() else np.zeros(0, dtype=np.int64)
    owner_mask = np.repeat(np.arange(k), sizes)
    order = np.argsort(owner_point, kind="stable")
    owner_point, owner_mask = owner_point[order], owner_mask[order]

    def masks_at(point: int) -> set[int]:
        lo, hi = np.searchsorted(owner_point, point, "left"), np.searchsorted(owner_point, point, "right")
        return set(owner_mask[lo:hi].tolist())

    seeded_by = [masks_at(int(s)) for s in seeds]
    for i in range(k):
        for j in seeded_by[i]:
            if j != i and i in seeded_by[j]:
                parent[root(i)] = root(j)

    #count points shared by each pair of masks: entries d apart in the sorted list with the same point
    pair_codes = []
    for d in range(1, k):
        if len(owner_point) <= d:
            break
        same = owner_point[d:] == owner_point[:-d]
        if not same.any():
            break
        a, b = owner_mask[:-d][same], owner_mask[d:][same]
        pair_codes.append(np.minimum(a, b) * k + np.maximum(a, b))
    if pair_codes:
        codes, shared = np.unique(np.concatenate(pair_codes), return_counts=True)
        for code, count in zip(codes.tolist(), shared.tolist()):
            a, b = divmod(code, k)
            if count / max(1, min(sizes[a], sizes[b])) >= containment:
                parent[root(a)] = root(b)

    best = np.full(n_points, -np.inf, dtype=np.float32)
    labels = np.full(n_points, -1, dtype=np.int64)
    for i in range(k):
        better = confidences[i] > best[sets[i]]
        best[sets[i][better]] = confidences[i][better]
        labels[sets[i][better]] = root(i)

    #renumber from 0 and drop anything too small to be a tree
    out = np.full(n_points, -1, dtype=np.int32)
    ids, inverse, counts = np.unique(labels[labels >= 0], return_inverse=True, return_counts=True)
    new_ids = np.where(counts >= min_points, np.cumsum(counts >= min_points) - 1, -1)
    out[labels >= 0] = new_ids[inverse]
    return out


def segment_tile(model, xyz: np.ndarray, click_groups: list[np.ndarray], device: str = "cuda", **crop_settings) -> np.ndarray:
    if not click_groups:
        return np.full(len(xyz), -1, dtype=np.int32)
    return fuse_masks(len(xyz), *predict_click_masks(model, xyz, click_groups, device=device, **crop_settings))
