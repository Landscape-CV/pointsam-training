"""
scoring.py - instance F1 at IoU 0.5, with one prediction matched to one tree.

Every tree in the whole tile counts, including trees that ground removal or the model lost completely.
Kevin's instance benchmark can miss these (a dropped tree never becomes a false negative), so here the
list of trees and their sizes comes from the full tile, before any points are removed.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def tree_sizes(ids: np.ndarray, min_points: int = 200) -> dict[int, int]:
    """Trees to score, from the full tile. Very small fragments are too small to judge fairly."""
    trees, counts = np.unique(ids[ids > 0], return_counts=True)
    return {int(t): int(c) for t, c in zip(trees, counts) if c >= min_points}


def match_instances(work_ids: np.ndarray, pred: np.ndarray, sizes: dict[int, int], iou_threshold: float = 0.5,
                    ignore_non_tree_preds: bool = False) -> dict:
    """Pair each tree with at most one prediction. This is the matching behind the F1 score, returned whole
    so a picture can colour the same pairs the score counted.

    matches maps a tree to (prediction, IoU) for pairs at or above the threshold. best_per_tree is every tree's
    best IoU with any prediction, 0 if nothing overlaps it. ignored lists predictions mostly on non-tree points.
    """
    pred_ids, pred_counts = np.unique(pred[pred >= 0], return_counts=True)
    ignored = []
    if ignore_non_tree_preds and len(pred_ids):
        keep = []
        for pid in pred_ids:
            if (work_ids[pred == pid] <= 0).mean() > 0.5:
                ignored.append(int(pid))
            else:
                keep.append(pid)
        pred_counts = pred_counts[np.isin(pred_ids, keep)]
        pred_ids = np.array(keep, dtype=pred_ids.dtype)

    tree_ids = np.array(sorted(sizes))
    n_trees, n_preds = len(tree_ids), len(pred_ids)
    matches = {}
    best_per_tree = np.zeros(n_trees)
    if n_trees and n_preds:
        tree_index = {t: i for i, t in enumerate(tree_ids.tolist())}
        pred_index = {p: j for j, p in enumerate(pred_ids.tolist())}
        iou = np.zeros((n_trees, n_preds))
        both = (pred >= 0) & (work_ids > 0)
        pairs, overlaps = np.unique(np.column_stack([work_ids[both], pred[both]]), axis=0, return_counts=True)
        for (tree, pid), inter in zip(pairs.tolist(), overlaps.tolist()):
            if tree in tree_index and pid in pred_index:
                i, j = tree_index[tree], pred_index[pid]
                iou[i, j] = inter / (sizes[tree] + pred_counts[j] - inter)
        best_per_tree = iou.max(axis=1)
        rows, cols = linear_sum_assignment(-iou)
        matches = {int(tree_ids[r]): (int(pred_ids[c]), float(iou[r, c])) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold}
    return {"tree_ids": tree_ids, "pred_ids": pred_ids, "matches": matches, "best_per_tree": best_per_tree, "ignored": ignored}


def score_instances(work_ids: np.ndarray, pred: np.ndarray, sizes: dict[int, int], iou_threshold: float = 0.5,
                    ignore_non_tree_preds: bool = False) -> dict:
    """work_ids and pred line up on the points kept after ground removal. pred is -1 for unassigned.

    ignore_non_tree_preds skips predictions mostly on non-tree points, instead of counting them as false
    positives. That is useful where unlabelled understory could be a correct find.
    """
    match = match_instances(work_ids, pred, sizes, iou_threshold, ignore_non_tree_preds)
    n_trees, n_preds = len(match["tree_ids"]), len(match["pred_ids"])
    matched = [iou for _, iou in match["matches"].values()]
    tp = len(matched)
    precision = tp / n_preds if n_preds else 0.0
    recall = tp / n_trees if n_trees else 0.0
    #missed trees count as 0 in mean_tree_iou, so they pull the average down
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "recall": recall, "tp": tp, "fp": n_preds - tp, "fn": n_trees - tp,
            "trees": n_trees, "predictions": n_preds, "preds_ignored": len(match["ignored"]),
            "mean_matched_iou": float(np.mean(matched)) if matched else float("nan"),
            "mean_tree_iou": float(match["best_per_tree"].mean()) if n_trees else 0.0}


def score_raw_masks(work_ids: np.ndarray, masks: list, sizes: dict[int, int], iou_threshold: float = 0.5) -> dict:
    """Score the per-click masks before they are merged: each tree's best IoU with any single mask.

    Comparing raw_recall with the recall after merging shows what merging costs. best maps a tree to
    (mask number, IoU), with mask -1 where no mask touches the tree.
    """
    best = {tree: (-1, 0.0) for tree in sizes}
    for number, mask in enumerate(masks):
        trees, inter = np.unique(work_ids[mask], return_counts=True)
        for tree, shared in zip(trees.tolist(), inter.tolist()):
            if tree in best:
                iou = shared / (sizes[tree] + len(mask) - shared)
                if iou > best[tree][1]:
                    best[tree] = (number, iou)
    ious = np.array([iou for _, iou in best.values()])
    return {"raw_masks": len(masks), "raw_trees_found": int((ious >= iou_threshold).sum()),
            "raw_recall": float((ious >= iou_threshold).mean()) if len(ious) else 0.0,
            "raw_mean_tree_iou": float(ious.mean()) if len(ious) else 0.0, "best": best}


def click_quality(work_xyz: np.ndarray, work_ids: np.ndarray, clicks: list[np.ndarray], sizes: dict[int, int],
                  max_distance: float = 0.3) -> dict:
    """How many scored trees have a first click on them, and how many clicks land on any tree."""
    from scipy.spatial import cKDTree

    if not clicks:
        return {"clicks": 0, "trees_hit": 0, "click_recall": 0.0, "clicks_on_tree": 0.0}
    distance, nearest = cKDTree(work_xyz).query(np.stack([c[0] for c in clicks]))
    labels = np.where(distance < max_distance, work_ids[nearest], -1)
    hit = set(labels[labels > 0].tolist()) & set(sizes)
    return {"clicks": len(clicks), "trees_hit": len(hit), "click_recall": len(hit) / max(len(sizes), 1),
            "clicks_on_tree": float((labels > 0).mean())}


def funnel(work_xyz: np.ndarray, work_ids: np.ndarray, sizes: dict[int, int], clicks: list[np.ndarray], raw: dict, match: dict) -> list[dict]:
    """How many scored trees are left after each stage. Each row is counted on its own, so a tree can be matched
    without its own click when a neighbour's mask covers it."""
    counts = [("Trees to find", len(sizes)), ("Has a click on it", click_quality(work_xyz, work_ids, clicks, sizes)["trees_hit"]),
              ("Has a good raw mask (before merging)", raw["raw_trees_found"]), ("Matched after merging", len(match["matches"]))]
    return [{"stage": stage, "trees": n, "share": round(n / max(len(sizes), 1), 3)} for stage, n in counts]
