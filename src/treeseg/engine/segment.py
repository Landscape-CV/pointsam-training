"""
segment.py - turn clicks into tree labels for a whole tile.

Ported from Kevin Hu's SegmenterPointSAM (local mode) and fuse_stem_masks. Each click runs in a 12 m crop
around it, so the model sees what it trained on. The per-click masks are then merged into one label per point.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree


def predict_click_masks(model, xyz: np.ndarray, click_groups: list[np.ndarray], crop_size: float = 12.0, crop_stride: float = 4.0,
                        crop_points: int = 10000, device: str = "cuda", seed: int = 0, negative_clicks: int = 0,
                        negative_radius: float = 6.0, centre_on_click: bool = False, pick: str = "score",
                        truth_ids: np.ndarray | None = None) -> tuple[list, list, list, list]:
    """Run the model. Returns, per click group: mask point indices, their logits, a score, and the seed point.

    With negative_clicks, each tree's prompt also gets up to that many "not this" clicks: the first clicks of the
    nearest other trees within negative_radius metres (measured from above). They tell the model where to stop.

    centre_on_click gives every click its own crop with the click in the middle. Otherwise clicks share crops on a
    grid, and a click can sit up to crop_stride / 2 from its crop's centre.

    pick says which of the model's 3 candidate masks is kept: "score" is the one the model rates highest, as in
    real use. "first" is always candidate 0. "truth" is the one that best overlaps the true tree under the click
    (measured on the crop's sampled points, from truth_ids), which shows what a perfect choice would be worth."""
    if pick not in ("score", "first", "truth") or (pick == "truth" and truth_ids is None):
        raise ValueError(f"pick must be score, first or truth (with truth_ids), got {pick!r}")
    rng = np.random.default_rng(seed)
    first_clicks = np.stack([group[0] for group in click_groups]) if click_groups else np.zeros((0, 3))
    click_tree = cKDTree(first_clicks[:, :2]) if negative_clicks and len(first_clicks) > 1 else None
    point_tree = cKDTree(xyz)
    seed_points = point_tree.query(first_clicks)[1] if len(first_clicks) else np.zeros(0, dtype=np.int64)
    xy_tree = cKDTree(xyz[:, :2])
    origin = xyz[:, :2].min(axis=0)
    half = crop_size / 2

    if centre_on_click:
        crops = [(group[0, :2], [g]) for g, group in enumerate(click_groups)]
    else:
        #clicks sharing a stride cell share one crop, which keeps a click within stride/2 of its crop centre
        cells: dict[tuple, list[int]] = {}
        for g, group in enumerate(click_groups):
            cells.setdefault(tuple(np.floor((group[0, :2] - origin) / crop_stride).astype(int).tolist()), []).append(g)
        crops = [(origin + (np.asarray(cell) + 0.5) * crop_stride, group_ids) for cell, group_ids in cells.items()]

    masks, logits, scores, seeds = [], [], [], []
    for centre, group_ids in crops:
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

        #each tree's own clicks are positive (1). Its neighbours' clicks, if asked for, are negative (0)
        own, others = [], []
        for g in group_ids:
            own.append(click_groups[g])
            near = np.zeros((0, 3))
            if click_tree is not None:
                distance, index = click_tree.query(first_clicks[g, :2], k=min(negative_clicks + 1, len(first_clicks)), distance_upper_bound=negative_radius)
                index = index[np.isfinite(distance) & (index != g)][:negative_clicks]
                #a click outside this crop would be clipped onto its edge, so only neighbours inside it are used
                near = first_clicks[index][(np.abs(first_clicks[index, :2] - centre) <= half).all(axis=1)]
            others.append(near)
        #different click counts are padded by repeating a group's first click, which stays positive
        most = max(len(a) + len(b) for a, b in zip(own, others))
        padded = np.stack([np.vstack([a, b, np.repeat(a[:1], most - len(a) - len(b), 0)]) for a, b in zip(own, others)])
        prompts = np.clip((padded - mean) / radius, -1.0, 1.0).astype(np.float32)
        prompt_labels = np.ones(prompts.shape[:2], dtype=np.int64)
        for k, (a, b) in enumerate(zip(own, others)):
            prompt_labels[k, len(a):len(a) + len(b)] = 0
        with torch.no_grad():
            out_masks, out_scores = model.predict_masks(coords, features, torch.from_numpy(prompts).to(device),
                                                        torch.from_numpy(prompt_labels).to(device), prompt_masks=None, multimask_output=True)
        #of the 3 candidates keep the one the model rates highest, as inference has no answer to check against
        best = torch.argmax(out_scores, dim=1)
        if pick == "first":
            best = torch.zeros_like(best)
        elif pick == "truth":
            candidates = (out_masks > 0).cpu().numpy()
            for k, g in enumerate(group_ids):
                tree = truth_ids[seed_points[g]]
                if tree > 0:  # a click that is not on a tree has no right answer, so the model's choice stands
                    target = truth_ids[sample] == tree
                    overlap = (candidates[k] & target).sum(axis=1) / np.maximum((candidates[k] | target).sum(axis=1), 1)
                    best[k] = int(overlap.argmax())
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
            seeds.append(int(seed_points[g]))
    return masks, logits, scores, seeds


def fuse_masks(n_points: int, masks: list, logits: list, scores: list, seeds: list, containment: float = 0.8, min_points: int = 50,
               return_groups: bool = False):
    """One label per point. Masks are the same tree if each contains the other's seed, or one sits at least
    `containment` inside the other. A point claimed by several trees goes to the highest logit.

    With return_groups, also returns each mask's final label: masks sharing a label were joined, and -1 means
    the mask was dropped (too small, or every point went to a more confident mask)."""
    k = len(masks)
    if k == 0:
        empty = np.full(n_points, -1, dtype=np.int32)
        return (empty, np.zeros(0, dtype=np.int32)) if return_groups else empty
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
    if not return_groups:
        return out
    final = dict(zip(ids.tolist(), new_ids.tolist()))
    return out, np.array([final.get(root(i), -1) for i in range(k)], dtype=np.int32)


def click_candidates(model, xyz: np.ndarray, group: np.ndarray, crop_size: float = 12.0, crop_points: int = 10000,
                     device: str = "cuda", seed: int = 0) -> dict:
    """One click group in a crop centred on it: the sampled points the model sees, its 3 candidate masks and
    their scores. For looking at a single click; predict_click_masks is what scores a tile."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero((np.abs(xyz[:, :2] - group[0, :2]) <= crop_size / 2).all(axis=1))
    sample = idx if len(idx) <= crop_points else rng.choice(idx, crop_points, replace=False)
    points = xyz[sample]
    mean = points.mean(axis=0)
    radius = float(np.linalg.norm(points - mean, axis=1).max()) or 1.0
    coords = torch.from_numpy(((points - mean) / radius).astype(np.float32)).unsqueeze(0).to(device)
    prompts = torch.from_numpy(np.clip((group - mean) / radius, -1.0, 1.0).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        out_masks, out_scores = model.predict_masks(coords, torch.zeros_like(coords), prompts,
                                                    torch.ones(prompts.shape[:2], dtype=torch.int64, device=device),
                                                    prompt_masks=None, multimask_output=True)
    scores = out_scores[0].float().cpu().numpy()
    return {"crop_index": idx, "sample_index": sample, "masks": (out_masks[0] > 0).cpu().numpy(), "scores": scores,
            "picked": int(scores.argmax()), "click": group}


def save_masks(path: str, clicks: list[np.ndarray], masks: list, logits: list, scores: list, seeds: list, keep: np.ndarray | None = None) -> None:
    """Everything predict_click_masks returned, plus the clicks, so merging can be replayed later without a GPU.

    keep marks which points of the whole tile were left after ground removal. Masks are point numbers among those,
    and ground removal differs slightly between machines, so a replay must use this instead of removing ground again."""
    rows = [np.column_stack([np.full(len(c), g), c]) for g, c in enumerate(clicks)]
    keep = np.zeros(0, dtype=bool) if keep is None else keep
    np.savez_compressed(path, clicks=np.vstack(rows) if rows else np.zeros((0, 4)), keep=np.packbits(keep), tile_points=len(keep),
                        lengths=np.array([len(m) for m in masks], dtype=np.int64),
                        points=np.concatenate(masks).astype(np.int32) if masks else np.zeros(0, dtype=np.int32),
                        logits=np.concatenate(logits).astype(np.float16) if logits else np.zeros(0, dtype=np.float16),
                        scores=np.array(scores, dtype=np.float32), seeds=np.array(seeds, dtype=np.int64))


def load_masks(path: str) -> dict:
    """Back to lists, in the order fuse_masks takes them: masks, logits, scores, seeds. Clicks come back grouped."""
    saved = np.load(path)
    cuts = np.cumsum(saved["lengths"])[:-1]
    clicks = saved["clicks"]
    split = (lambda values: np.split(values, cuts)) if len(saved["lengths"]) else (lambda values: [])
    keep = np.unpackbits(saved["keep"])[: int(saved["tile_points"])].astype(bool) if "keep" in saved and int(saved["tile_points"]) else None
    return {"clicks": [clicks[clicks[:, 0] == g, 1:] for g in np.unique(clicks[:, 0])], "keep": keep,
            "masks": split(saved["points"].astype(np.int64)), "logits": split(saved["logits"].astype(np.float32)),
            "scores": saved["scores"].tolist(), "seeds": saved["seeds"].tolist()}


def segment_tile(model, xyz: np.ndarray, click_groups: list[np.ndarray], device: str = "cuda", **crop_settings) -> np.ndarray:
    if not click_groups:
        return np.full(len(xyz), -1, dtype=np.int32)
    return fuse_masks(len(xyz), *predict_click_masks(model, xyz, click_groups, device=device, **crop_settings))
