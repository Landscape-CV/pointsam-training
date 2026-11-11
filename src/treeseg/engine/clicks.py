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


#TreeX's stock TLS settings find no trunks on our 5 cm tiles, so these relax its density-sensitive trunk checks.
#They are Kevin Hu's settings for sparser scans, and reached 50-65% of trees on the val tiles.
TREEX_RELAXED = {
    "num_workers": 1,  # more workers stalled on the Mac
    "stem_search_dbscan_2d_min_points": 15,
    "stem_search_dbscan_2d_eps": 0.08,
    "stem_search_dbscan_3d_min_points": 8,
    "stem_search_dbscan_3d_eps": 0.15,
    "stem_search_min_cluster_points": 40,
    "stem_search_min_cluster_height": 0.6,
    "stem_search_min_cluster_intensity": None,
    "stem_search_voxel_size": 0.04,
    "stem_search_circle_fitting_min_points": 8,
    "stem_search_circle_fitting_min_fitting_score": 20.0,
    "stem_search_circle_fitting_min_completeness_idx": 0.15,
    "stem_search_circle_fitting_max_std_diameter": 0.15,
    "stem_search_circle_fitting_layer_height": 0.35,
    "stem_search_circle_fitting_num_layers": 8,
    "stem_search_circle_fitting_layer_overlap": 0.05,
}


def clicks_at_positions(xyz: np.ndarray, positions: np.ndarray, height: float = 1.3, radius: float = 0.5) -> list[np.ndarray]:
    """Turn trunk positions (x, y) into clicks: the nearby point closest to `height` above the ground."""
    from scipy.spatial import cKDTree

    heights = BaseSurface(xyz).height(xyz)
    xy_tree = cKDTree(xyz[:, :2])
    clicks = []
    for position in positions:
        near = np.asarray(xy_tree.query_ball_point(position, r=radius))
        if len(near):
            cost = np.abs(heights[near] - height) + 0.5 * np.hypot(*(xyz[near, :2] - position).T)
            clicks.append(xyz[near[np.argmin(cost)]].reshape(1, 3))
    return clicks


def treex_segment(full_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The whole TreeX algorithm: trunk detection, then growing each crown out from its trunk. It removes the
    ground itself, so it gets the whole tile. Returns a tree label per point (-1 for none) and trunk positions (x, y)."""
    from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS

    offset = full_xyz.mean(axis=0)
    labels, positions, _ = TreeXAlgorithm(**{**dict(TreeXPresetTLS()), **TREEX_RELAXED})((full_xyz - offset).astype(np.float64))
    return np.asarray(labels).astype(np.int32), np.asarray(positions).reshape(-1, 2) + offset[:2]


def treex_clicks(xyz: np.ndarray, full_xyz: np.ndarray) -> list[np.ndarray]:
    """TreeX trunk detection only: its tree labels are not used, and each trunk becomes one click."""
    return clicks_at_positions(xyz, treex_segment(full_xyz)[1])


#every click method takes (points without ground, labels, whole tile) and returns a list of click arrays. To add one: write the function,
#add it here, and name it under click_methods in a config. Nothing downstream needs to change.
#Only "perfect" uses the labels. It's there to show how good masks can be.
CLICK_METHODS = {
    "grid": lambda xyz, ids, full_xyz: grid_clicks(xyz),
    "stems": lambda xyz, ids, full_xyz: stem_clicks(xyz),
    "treex": lambda xyz, ids, full_xyz: treex_clicks(xyz, full_xyz),
    "perfect": lambda xyz, ids, full_xyz: perfect_clicks(xyz, ids),
}


def make_clicks(method: str, xyz: np.ndarray, ids: np.ndarray, full_xyz: np.ndarray | None = None) -> list[np.ndarray]:
    if method not in CLICK_METHODS:
        raise ValueError(f"unknown click method {method!r}, choose from {sorted(CLICK_METHODS)}")
    return CLICK_METHODS[method](xyz, ids, xyz if full_xyz is None else full_xyz)
