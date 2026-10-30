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


def score_instances(work_ids: np.ndarray, pred: np.ndarray, sizes: dict[int, int], iou_threshold: float = 0.5,
                    ignore_non_tree_preds: bool = False) -> dict:
    """work_ids and pred line up on the points kept after ground removal. pred is -1 for unassigned.

    ignore_non_tree_preds skips predictions mostly on non-tree points, instead of counting them as false
    positives. That is useful where unlabelled understory could be a correct find.
    """
    pred_ids, pred_counts = np.unique(pred[pred >= 0], return_counts=True)
    ignored = 0
    if ignore_non_tree_preds and len(pred_ids):
        keep = []
        for pid in pred_ids:
            if (work_ids[pred == pid] <= 0).mean() > 0.5:
                ignored += 1
            else:
                keep.append(pid)
        pred_counts = pred_counts[np.isin(pred_ids, keep)]
        pred_ids = np.array(keep, dtype=pred_ids.dtype)

    tree_ids = np.array(sorted(sizes))
    n_trees, n_preds = len(tree_ids), len(pred_ids)
    matched = []
    #every tree's best IoU with any prediction, 0 if nothing overlaps it, so missed trees pull the average down
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
        matched = [iou[r, c] for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]

    tp = len(matched)
    precision = tp / n_preds if n_preds else 0.0
    recall = tp / n_trees if n_trees else 0.0
    return {"f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "recall": recall, "tp": tp, "fp": n_preds - tp, "fn": n_trees - tp,
            "trees": n_trees, "predictions": n_preds, "preds_ignored": ignored,
            "mean_matched_iou": float(np.mean(matched)) if matched else float("nan"),
            "mean_tree_iou": float(best_per_tree.mean()) if n_trees else 0.0}


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
