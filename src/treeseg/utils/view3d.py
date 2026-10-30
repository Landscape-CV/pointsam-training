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
