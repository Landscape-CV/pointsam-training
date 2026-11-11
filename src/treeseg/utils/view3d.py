"""
view3d.py - see clicks and predictions in a notebook.

The 3D views use Plotly: drag to rotate, scroll to zoom, click a legend entry to hide it. They draw a sample
of points, because a browser cannot show millions. Click markers are never sampled.
"""
import numpy as np

TREE_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NOT_TREE, HIT, MISS = "#c3c2b7", "#008300", "#e34948"


def colour_by_id(ids: np.ndarray) -> list[str]:
    """Each id keeps one colour, with neighbouring ids spread apart. Non-tree points are grey."""
    return [NOT_TREE if i <= 0 else TREE_COLOURS[(int(i) * 5) % len(TREE_COLOURS)] for i in ids]


def _sample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).choice(n, min(max_points, n), replace=False)


def click_status(xyz: np.ndarray, ids: np.ndarray, clicks: list[np.ndarray], max_distance: float = 0.3):
    """For each click: the tree it landed on (or -1), and how far it is from the nearest point."""
    from scipy.spatial import cKDTree

    distance, nearest = cKDTree(xyz).query(np.stack([c[0] for c in clicks]))
    return np.where(distance < max_distance, ids[nearest], -1), distance


def plot_clicks_3d(xyz: np.ndarray, ids: np.ndarray, clicks: list[np.ndarray], title: str, max_points: int = 120000):
    """True trees in colour, grey for non-tree. Green diamonds are clicks on a tree, red crosses are misses."""
    import plotly.graph_objects as go

    picked = _sample(len(xyz), max_points)
    labels, _ = click_status(xyz, ids, clicks)
    points = np.stack([c[0] for c in clicks])
    hit = labels > 0
    fig = go.Figure(go.Scatter3d(x=xyz[picked, 0], y=xyz[picked, 1], z=xyz[picked, 2], mode="markers", name="points (by true tree)",
                                 marker={"size": 1.2, "color": colour_by_id(ids[picked])}))
    for mask, name, color, symbol in ((hit, "click on a tree", HIT, "diamond"), (~hit, "click missed", MISS, "x")):
        fig.add_trace(go.Scatter3d(x=points[mask, 0], y=points[mask, 1], z=points[mask, 2], mode="markers", name=f"{name} ({mask.sum()})",
                                   marker={"size": 7, "color": color, "symbol": symbol, "line": {"color": "white", "width": 1}}))
    fig.update_layout(title=title, height=680, scene={"aspectmode": "data"}, legend={"itemsizing": "constant"})
    return fig


def plot_click_map(xyz: np.ndarray, ids: np.ndarray, clicks: list[np.ndarray], title: str, min_tree_points: int = 200, spacing: float | None = None, ax=None):
    """Top-down: each tree's trunk position, marked by whether any click landed on it, plus every click."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7), dpi=130)
    labels, _ = click_status(xyz, ids, clicks)
    hit_trees = set(labels[labels > 0].tolist())
    trees, counts = np.unique(ids[ids > 0], return_counts=True)
    trees = trees[counts >= min_tree_points]
    bases = []
    for tree in trees:
        points = xyz[ids == tree]
        bases.append(np.median(points[points[:, 2] <= np.quantile(points[:, 2], 0.1), :2], axis=0))
    bases = np.array(bases)
    found = np.isin(trees, list(hit_trees))
    ax.scatter(*bases[found].T, s=60, facecolors="none", edgecolors=HIT, linewidths=1.5, label=f"tree with a click ({found.sum()})")
    ax.scatter(*bases[~found].T, s=60, facecolors="none", edgecolors=MISS, linewidths=1.5, label=f"tree with no click ({(~found).sum()})")
    points = np.stack([c[0] for c in clicks])
    ax.scatter(points[labels > 0, 0], points[labels > 0, 1], s=22, c=HIT, marker="D", linewidths=0, label=f"click on a tree ({(labels > 0).sum()})")
    ax.scatter(points[labels <= 0, 0], points[labels <= 0, 1], s=22, c=MISS, marker="x", linewidths=1.2, label=f"click missed ({(labels <= 0).sum()})")
    if spacing:
        x0, y0 = xyz[:, :2].min(axis=0)
        ax.set_xticks(np.arange(x0, xyz[:, 0].max() + spacing, spacing), minor=True)
        ax.set_yticks(np.arange(y0, xyz[:, 1].max() + spacing, spacing), minor=True)
        ax.grid(which="minor", color="#e6e5e0", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    return ax


def plot_stem_steps(xyz: np.ndarray, height: np.ndarray, stem: dict, radius: float = 2.5, title: str = ""):
    """One detected trunk, seen from the side and from above: the search slice and where the clicks went."""
    import matplotlib.pyplot as plt

    near = np.hypot(*(xyz[:, :2] - stem["xy"]).T) < radius
    low = near & (height < 8)
    in_slice = low & (height >= 0.3) & (height <= 2.5)
    fig, (side, top) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=130)
    for ax, u, v, xlabel, ylabel in ((side, 0, 2, "x (m)", "z (m)"), (top, 0, 1, "x (m)", "y (m)")):
        ax.scatter(xyz[low & ~in_slice, u], xyz[low & ~in_slice, v], s=1, c=NOT_TREE, linewidths=0, label="nearby points")
        ax.scatter(xyz[in_slice, u], xyz[in_slice, v], s=1.5, c="#2a78d6", linewidths=0, label="search slice, 0.3-2.5 m")
        ax.scatter(stem["click"][u], stem["click"][v], s=70, c=HIT, marker="D", edgecolors="white", label="stem click")
        if stem["upper"] is not None:
            ax.scatter(stem["upper"][u], stem["upper"][v], s=70, c="#eb6834", marker="D", edgecolors="white", label="upper click")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    side.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.suptitle(title, x=0.05, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_compare_3d(xyz: np.ndarray, truth: np.ndarray, pred: np.ndarray, title: str, max_points: int = 80000):
    """True trees and predicted trees side by side, on the same sampled points. Colours are per panel."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    picked = _sample(len(xyz), max_points)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("True trees", "Predicted trees"))
    for col, labels in ((1, truth), (2, np.where(pred >= 0, pred + 1, -1))):
        fig.add_trace(go.Scatter3d(x=xyz[picked, 0], y=xyz[picked, 1], z=xyz[picked, 2], mode="markers", showlegend=False,
                                   marker={"size": 1.2, "color": colour_by_id(labels[picked])}), row=1, col=col)
    fig.update_layout(title=title, height=620, scene={"aspectmode": "data"}, scene2={"aspectmode": "data"})
    return fig


def plot_click_kinds(item: dict, positions: dict, mask_index: int = 0, title: str = ""):
    """One target tree from the side and from above, with each kind of training click marked."""
    import matplotlib.pyplot as plt

    coords, truth = item["coords"].numpy(), item["gt_masks"][mask_index].numpy()
    styles = {"stem": ("#008300", "D", "stem click"), "upper": ("#6250d6", "s", "upper stem click"), "random": ("#e34948", "X", "random point on tree")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for ax, (u, v), xlabel, ylabel in ((axes[0], (0, 2), "x", "height"), (axes[1], (0, 1), "x", "y")):
        ax.scatter(coords[~truth, u], coords[~truth, v], s=1, c=NOT_TREE, linewidths=0, label="other points")
        ax.scatter(coords[truth, u], coords[truth, v], s=2, c="#eb6834", linewidths=0, label="target tree")
        for name, point in positions.items():
            color, marker, label = styles[name]
            ax.scatter(point[u], point[v], s=110, c=color, marker=marker, edgecolors="white", linewidths=1, label=label, zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, markerscale=1)
    fig.suptitle(title, x=0.05, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_mask_candidates(result: dict, title: str = ""):
    """The true tree, then the 3 masks the model returned for one click, side view. Green title: the model's pick."""
    import matplotlib.pyplot as plt

    coords, truth = result["coords"], result["truth"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), dpi=110, sharey=True)
    panels = [("True tree", truth, None)] + [(f"Mask {i + 1}: IoU {result['ious'][i]:.2f}, model score {result['scores'][i]:.2f}", result["masks"][i], i) for i in range(3)]
    for ax, (label, mask, index) in zip(axes, panels):
        ax.scatter(coords[~mask, 0], coords[~mask, 2], s=0.8, c=NOT_TREE, linewidths=0)
        ax.scatter(coords[mask, 0], coords[mask, 2], s=1.6, c="#eb6834", linewidths=0)
        ax.scatter(result["click"][:, 0], result["click"][:, 2], s=110, c="#008300", marker="D", edgecolors="white", zorder=5)
        picked = index == result["picked"]
        ax.set_title(("MODEL'S PICK\n" if picked else "\n") + label, fontsize=10, color="#008300" if picked else "#0b0b0b", loc="left")
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


#outcome colours: a matched tree and its prediction share one colour, so red is kept for wrong predictions only
MATCH_COLOURS = [c for c in TREE_COLOURS if c != MISS]
MISSED_TREE, IGNORED_PRED = "#4d4d4d", "#d9c9a3"


def outcome_colours(ids: np.ndarray, pred: np.ndarray, match: dict) -> tuple[np.ndarray, np.ndarray]:
    """A colour per point for the truth and for the prediction, from match_instances: matched pairs share a colour,
    missed trees are dark, unmatched predictions red, ignored (understory) predictions tan, the rest grey."""
    truth_colour = {int(t): MISSED_TREE for t in match["tree_ids"]}
    pred_colour = {int(p): MISS for p in match["pred_ids"]} | {p: IGNORED_PRED for p in match["ignored"]}
    for k, (tree, (pid, _)) in enumerate(match["matches"].items()):
        truth_colour[tree] = pred_colour[pid] = MATCH_COLOURS[(k * 3) % len(MATCH_COLOURS)]
    return (np.array([truth_colour.get(int(i), NOT_TREE) for i in ids], dtype=object),
            np.array([pred_colour.get(int(i), NOT_TREE) for i in pred], dtype=object))


def _draw(ax, points: np.ndarray, colours, u: int, v: int, title: str, size: float = 0.6) -> None:
    ax.scatter(points[:, u], points[:, v], s=size, c=list(colours), linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("xyz"[u] + " (m)")
    ax.set_ylabel("xyz"[v] + " (m)")
    ax.spines[["top", "right"]].set_visible(False)


def plot_outcome(xyz: np.ndarray, ids: np.ndarray, pred: np.ndarray, match: dict, clicks: list[np.ndarray] | None = None, title: str = "",
                 slab: float = 6.0, max_points: int = 300000):
    """True trees next to predicted trees, from above and as a side slice through the middle of the tile.
    Colours follow outcome_colours, so the same colour on both sides is one matched tree."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    picked = _sample(len(xyz), max_points)
    picked = picked[np.argsort(xyz[picked, 2])]  # highest points drawn last, as seen from above
    points = xyz[picked]
    truth_colours, pred_colours = outcome_colours(ids[picked], pred[picked], match)
    middle = np.median(xyz[:, 1])
    in_slab = np.abs(points[:, 1] - middle) <= slab / 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=110, height_ratios=[1.6, 1])
    n_trees, n_matched = len(match["tree_ids"]), len(match["matches"])
    wrong = len(match["pred_ids"]) - n_matched
    _draw(axes[0, 0], points, truth_colours, 0, 1, f"True trees: {n_matched} of {n_trees} matched")
    _draw(axes[0, 1], points, pred_colours, 0, 1, f"Predicted trees: {n_matched} matched, {wrong} wrong, {len(match['ignored'])} on understory")
    _draw(axes[1, 0], points[in_slab], truth_colours[in_slab], 0, 2, f"True trees, side view of a {slab:g} m slice", size=1.2)
    _draw(axes[1, 1], points[in_slab], pred_colours[in_slab], 0, 2, "Predicted trees, the same slice", size=1.2)
    for ax in axes[0]:
        ax.axhspan(middle - slab / 2, middle + slab / 2, color="#0b0b0b", alpha=0.06, linewidth=0)
    if clicks:
        labels, _ = click_status(xyz, ids, clicks)
        first = np.stack([c[0] for c in clicks])
        axes[0, 1].scatter(first[labels > 0, 0], first[labels > 0, 1], s=16, c="#0b0b0b", marker="D", linewidths=0)
        axes[0, 1].scatter(first[labels <= 0, 0], first[labels <= 0, 1], s=18, c="#0b0b0b", marker="x", linewidths=1)
    dot = lambda colour, label, marker="o": Line2D([], [], marker=marker, linestyle="", color=colour, markersize=7, label=label)
    fig.legend(handles=[dot(MATCH_COLOURS[0], "matched (same colour on both sides)"), dot(MISSED_TREE, "true tree with no match"),
                        dot(MISS, "prediction matching no tree"), dot(IGNORED_PRED, "prediction on understory, not counted"),
                        dot(NOT_TREE, "not a tree / unassigned"), dot("#0b0b0b", "click on a tree", "D"), dot("#0b0b0b", "click that missed", "x")],
               frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    return fig


def plot_outcome_3d(xyz: np.ndarray, ids: np.ndarray, pred: np.ndarray, match: dict, title: str = "", max_points: int = 80000):
    """plot_outcome's colours in 3D, to rotate. Left is the truth, right is the prediction."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    picked = _sample(len(xyz), max_points)
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=("True trees", "Predicted trees"))
    for col, colours in enumerate(outcome_colours(ids[picked], pred[picked], match), start=1):
        fig.add_trace(go.Scatter3d(x=xyz[picked, 0], y=xyz[picked, 1], z=xyz[picked, 2], mode="markers", showlegend=False,
                                   marker={"size": 1.2, "color": list(colours)}), row=1, col=col)
    fig.update_layout(title=title, height=620, scene={"aspectmode": "data"}, scene2={"aspectmode": "data"})
    return fig


def plot_crop_sampling(xyz: np.ndarray, candidates: dict, title: str = ""):
    """Side view of one crop: every point in it, then the sample the model is given. The diamond is the click."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110, sharey=True)
    for ax, index, label in ((axes[0], candidates["crop_index"], "All points in the crop"), (axes[1], candidates["sample_index"], "What the model sees")):
        _draw(ax, xyz[index], [NOT_TREE if len(index) > 50000 else "#6b6a63"] * len(index), 0, 2, f"{label}: {len(index):,}", size=0.5 if len(index) > 50000 else 1.5)
        ax.scatter(candidates["click"][:, 0], candidates["click"][:, 2], s=110, c=HIT, marker="D", edgecolors="white", zorder=5)
    fig.suptitle(title, x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_merge(xyz: np.ndarray, masks: list, groups: np.ndarray, pred: np.ndarray, title: str = "", max_points: int = 300000):
    """Merging, from above: the raw masks one colour each, the same masks coloured by what merging did to them,
    then the final labels. Where masks overlap, the later one is drawn."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    owner = np.full(len(xyz), -1)
    for number, mask in enumerate(masks):
        owner[mask] = number
    shared = np.bincount(groups[groups >= 0], minlength=1)
    fate_colour = {"kept by itself": "#2a78d6", "joined with another mask": "#eda100", "dropped": MISS}
    fates = np.array(["dropped" if g < 0 else "joined with another mask" if shared[g] > 1 else "kept by itself" for g in groups], dtype=object)

    picked = _sample(len(xyz), max_points)
    picked = picked[np.argsort(xyz[picked, 2])]
    points, own = xyz[picked], owner[picked]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), dpi=110)
    _draw(axes[0], points, colour_by_id(np.where(own >= 0, own + 1, -1)), 0, 1, f"1. Raw masks: {len(masks)}, one per click")
    _draw(axes[1], points, [NOT_TREE if o < 0 else fate_colour[fates[o]] for o in own], 0, 1, "2. What merging does to each mask")
    _draw(axes[2], points, colour_by_id(np.where(pred[picked] >= 0, pred[picked] + 1, -1)), 0, 1, f"3. After merging: {len(np.unique(pred[pred >= 0]))} trees")
    axes[1].legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=f"{name} ({(fates == name).sum()})")
                            for name, c in fate_colour.items()], frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_tree_gallery(xyz: np.ndarray, ids: np.ndarray, pred: np.ndarray, match: dict, title: str = "", margin: float = 1.5):
    """Four trees from the side: the best match, a typical match, the nearest miss and the largest tree with nothing
    close. Each shows the part of the tree that was found, the part that was missed, and where the prediction spilled."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    trees, best = match["tree_ids"], match["best_per_tree"]
    matched = sorted(match["matches"], key=lambda t: match["matches"][t][1])
    missed = [int(t) for t in trees if int(t) not in match["matches"]]
    chosen = []
    if matched:
        chosen += [("Best match", matched[-1]), ("Typical match", matched[len(matched) // 2])]
    if missed:
        iou_of = dict(zip(trees.tolist(), best.tolist()))
        chosen.append(("Nearest miss", max(missed, key=lambda t: iou_of[t])))
        far = [t for t in missed if iou_of[t] < 0.1]
        if far:
            chosen.append(("Largest tree with nothing close", max(far, key=lambda t: (ids == t).sum())))
    if not chosen:
        return None

    fig, axes = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 6.5), dpi=110, squeeze=False)
    for ax, (label, tree) in zip(axes[0], chosen):
        is_tree = ids == tree
        on_tree = pred[is_tree]
        #the prediction to compare with: the matched one, else whichever covers most of the tree
        if tree in match["matches"]:
            pid = match["matches"][tree][0]
        else:
            values, counts = np.unique(on_tree[on_tree >= 0], return_counts=True)
            pid = int(values[counts.argmax()]) if len(values) else -2
        is_pred = pred == pid
        low, high = xyz[is_tree, :2].min(axis=0) - margin, xyz[is_tree, :2].max(axis=0) + margin
        near = ((xyz[:, :2] >= low) & (xyz[:, :2] <= high)).all(axis=1)
        for mask, colour, size in ((near & ~is_tree & ~is_pred, NOT_TREE, 0.4), (is_tree & ~is_pred, MISSED_TREE, 1.2),
                                   (near & is_pred & ~is_tree, MISS, 1.2), (is_tree & is_pred, "#1baf7a", 1.2)):
            ax.scatter(xyz[mask, 0], xyz[mask, 2], s=size, c=colour, linewidths=0, rasterized=True)
        inter, n_pred = (is_tree & is_pred).sum(), is_pred.sum()
        iou = inter / max(is_tree.sum() + n_pred - inter, 1)
        spill = f"{100 * (n_pred - inter) / n_pred:.0f}% of the prediction is outside it" if n_pred else "no prediction on it"
        ax.set_title(f"{label}: IoU {iou:.2f}\n{100 * inter / is_tree.sum():.0f}% of the tree found\n{spill}", loc="left", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("z (m)")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=name) for name, c in
                        (("tree, found", "#1baf7a"), ("tree, missed", MISSED_TREE), ("prediction outside the tree", MISS), ("other points", NOT_TREE))],
               frameon=False, fontsize=9, loc="lower center", ncol=4)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig
