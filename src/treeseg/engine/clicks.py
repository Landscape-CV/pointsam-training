"""
clicks.py - choose where to click without using labels, plus perfect clicks for comparison.

Ported unchanged in behaviour from Kevin Hu's promptable_instance.py, so the first run has a baseline.
All of these expect a tile with the ground already removed. Each click list item is an array (K, 3),
with K clicks on the same tree.
"""
import numpy as np


def grid_clicks(xyz: np.ndarray, spacing: float = 4.0, low_fraction: float = 0.15, min_points: int = 30) -> list[np.ndarray]:
    """Lowest points of the tile, cut into squares, one click at each square's median point."""
    low = xyz[xyz[:, 2] <= np.quantile(xyz[:, 2], low_fraction)]
    cells = np.floor((low[:, :2] - low[:, :2].min(axis=0)) / spacing).astype(int)
    clicks = []
    for cell in np.unique(cells, axis=0):
        points = low[(cells == cell).all(axis=1)]
        if len(points) >= min_points:
            clicks.append(np.median(points, axis=0).reshape(1, 3))
    return clicks


def perfect_clicks(xyz: np.ndarray, ids: np.ndarray, low_fraction: float = 0.10) -> list[np.ndarray]:
    """One click per labelled tree, from the true labels. Shows how good masks can be with ideal clicks."""
    clicks = []
    for tree in np.unique(ids[ids > 0]):
        points = xyz[ids == tree]
        stem = points[points[:, 2] <= np.quantile(points[:, 2], low_fraction)]
        clicks.append(np.median(stem, axis=0).reshape(1, 3))
    return clicks


class BaseSurface:
    """Estimated ground height on a 1 m grid. Works on clouds with the ground already removed."""

    def __init__(self, xyz: np.ndarray, cell: float = 1.0, window: int = 3, max_slope: float = 0.6):
        from scipy.ndimage import minimum_filter

        self.cell = cell
        self.x0, self.y0 = xyz[:, 0].min(), xyz[:, 1].min()
        ix = np.floor((xyz[:, 0] - self.x0) / cell).astype(np.int64)
        iy = np.floor((xyz[:, 1] - self.y0) / cell).astype(np.int64)
        self.nx, self.ny = int(ix.max()) + 1, int(iy.max()) + 1
        lowest = np.full((self.ny, self.nx), np.inf)
        np.minimum.at(lowest, (iy, ix), xyz[:, 2])
        #a canopy-only cell falls back on nearby low points, allowing for slope
        dtm = np.minimum(lowest, minimum_filter(lowest, size=2 * window + 1, mode="nearest") + max_slope * window * cell)
        for _ in range(64):
            empty = ~np.isfinite(dtm)
            if not empty.any():
                break
            dtm[empty] = minimum_filter(dtm, size=3, mode="nearest")[empty]
        self.dtm = dtm

    def height(self, xyz: np.ndarray) -> np.ndarray:
        """Height above the surface, using bilinear interpolation between grid cells."""
        fx = np.clip((xyz[:, 0] - self.x0) / self.cell - 0.5, 0, self.nx - 1)
        fy = np.clip((xyz[:, 1] - self.y0) / self.cell - 0.5, 0, self.ny - 1)
        i, j = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
        i1, j1 = np.minimum(i + 1, self.nx - 1), np.minimum(j + 1, self.ny - 1)
        tx, ty, d = fx - i, fy - j, self.dtm
        return xyz[:, 2] - ((d[j, i] * (1 - tx) + d[j, i1] * tx) * (1 - ty) + (d[j1, i] * (1 - tx) + d[j1, i1] * tx) * ty)


def detect_stems(xyz: np.ndarray, slice_low: float = 0.3, slice_high: float = 2.5, voxel: float = 0.05, eps: float = 0.1,
                 min_samples: int = 6, min_points: int = 40, min_vertical: float = 0.5, max_width: float = 1.0,
                 merge_distance: float = 1.5, click_height: float = 1.3, upper_height: float = 4.0) -> list[dict]:
    """Find trunks: clusters in a low height slice that are narrow and vertical. Largest first."""
    from scipy.spatial import cKDTree
    from sklearn.cluster import DBSCAN

    height = BaseSurface(xyz).height(xyz)
    in_slice = (height >= slice_low) & (height <= slice_high)
    if in_slice.sum() < min_points:
        return []
    points, heights = xyz[in_slice], height[in_slice]
    _, keep = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    points, heights = points[keep], heights[keep]
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    found = []
    for cluster in np.unique(labels[labels >= 0]):
        member = labels == cluster
        if member.sum() < min_points:
            continue
        width = np.ptp(points[member, :2], axis=0).max()
        if width > max_width or np.ptp(heights[member]) < min_vertical:
            continue
        found.append((int(member.sum()), np.median(points[member, :2], axis=0)))
    found.sort(key=lambda item: -item[0])

    #two clusters this close are treated as parts of one trunk
    kept: list[tuple[int, np.ndarray]] = []
    for size, centre in found:
        if all(np.hypot(*(centre - other)) >= merge_distance for _, other in kept):
            kept.append((size, centre))

    xy_tree = cKDTree(xyz[:, :2])

    def nearest_at(centre, target, radius):
        near = np.asarray(xy_tree.query_ball_point(centre, r=radius))
        if len(near) == 0:
            return None
        best = near[np.argmin(np.abs(height[near] - target) + 0.5 * np.hypot(*(xyz[near, :2] - centre).T))]
        return xyz[best] if abs(height[best] - target) < 1.0 else None

    stems = []
    for size, centre in kept:
        click = nearest_at(centre, click_height, 0.5)
        if click is not None:
            stems.append({"click": click, "upper": nearest_at(centre, upper_height, 0.8), "size": size, "xy": centre})
    return stems


def stem_clicks(xyz: np.ndarray, **settings) -> list[np.ndarray]:
    """One entry per trunk: a click about 1.3 m up, plus one about 4 m up when there is one."""
    return [np.stack([s["click"]] + ([s["upper"]] if s["upper"] is not None else [])) for s in detect_stems(xyz, **settings)]
