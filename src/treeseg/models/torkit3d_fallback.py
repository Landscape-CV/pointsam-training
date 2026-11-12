"""
torkit3d_fallback.py - plain-PyTorch stand-ins for the three torkit3d functions Point-SAM uses.

torkit3d ships only CUDA kernels, so Point-SAM cannot even be imported on a Mac or a CPU-only machine.
These versions run on any device (cpu, mps, cuda). They follow the CUDA kernels at Point-SAM commit 25f4fd9
(third_party/torkit3d/torkit3d/csrc/cuda/): same signatures, same return shapes, same starting point.

install_torkit3d() is called by build_point_sam. It leaves the real torkit3d alone when that can run, and
otherwise registers a stand-in package named torkit3d, so upstream Point-SAM files need no edits.

Checked against the real torkit3d on ICE (A40, 11 Nov 2026) with scripts/compare_fallback.py: identical masks and
scores, but point sampling is about 30x slower as a Python loop, so a crop takes 1.6x as long. See docs/no_cuda.md.
"""
import contextlib
import importlib
import sys
import types

import torch


def sample_farthest_points(points: torch.Tensor, num_samples: int, transpose: bool = False) -> torch.Tensor:
    """Farthest point sampling. points [B, N, 3] (or [B, 3, N] with transpose) -> indices [B, num_samples], int64.

    Like the CUDA kernel, the first pick is always index 0, and each next pick is the point furthest from
    everything picked so far. If no point is further than 0 away, the last pick is repeated.
    Ties go to the lowest index. The kernel breaks ties by thread instead, which only matters for duplicated points."""
    if transpose:
        points = points.transpose(1, 2)
    b, n, dim = points.shape
    if dim != 3:
        raise ValueError(f"points must be [B, N, 3], got {tuple(points.shape)}")
    if not 0 < num_samples <= n:
        raise ValueError(f"num_samples must be in 1..{n}, got {num_samples}")
    #on mps the loop below pays a Metal call per pick and is about 6x slower than the CPU (measured, 10,000 points),
    #so it runs on the CPU and the indices are moved back. That also gives cpu and mps the same picks.
    device = points.device
    points = points.detach().cpu() if device.type == "mps" else points.detach()
    index = torch.zeros(b, num_samples, dtype=torch.long, device=points.device)
    min_dist = torch.full((b, n), float("inf"), dtype=points.dtype, device=points.device)
    selected = torch.zeros(b, dtype=torch.long, device=points.device)
    batch = torch.arange(b, device=points.device)
    x, y, z = points.unbind(-1)
    for i in range(1, num_samples):
        last = points[batch, selected]  # [B, 3]
        dx, dy, dz = x - last[:, 0:1], y - last[:, 1:2], z - last[:, 2:3]
        #summed in the kernel's order (x, then y, then z), to round the same way
        min_dist = torch.minimum(min_dist, dx * dx + dy * dy + dz * dz)
        furthest, candidate = min_dist.max(dim=1)
        selected = torch.where(furthest > 0, candidate, selected)
        index[:, i] = selected
    return index.to(device)


def batch_index_select(input: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """torch.index_select with a different index per batch item. index is [B, N], or [B] to drop that dimension.

    The real one is already plain PyTorch (torkit3d/nn/functional.py). This is the same code."""
    squeeze = index.dim() == 1
    if squeeze:
        index = index.unsqueeze(1)
    if index.dim() != 2:
        raise ValueError(f"index must be 1-dim or 2-dim, got {index.dim()}-dim")
    if input.size(0) != index.size(0):
        raise ValueError(f"mismatched batch size: {input.size(0)} vs {index.size(0)}")
    view = [1] * input.dim()
    view[0], view[dim] = index.size(0), index.size(1)
    shape = list(input.shape)
    shape[dim] = -1
    out = torch.gather(input, dim, index.view(view).expand(shape))
    #the original squeezes dimension 1 whatever dim is; kept as is, since Point-SAM only uses dim=1
    return out.squeeze(1) if squeeze else out


def _nearest(a: torch.Tensor, b: torch.Tensor, chunk: int) -> torch.Tensor:
    """For each point of a [B, N1, 3], the index of its nearest point in b [B, N2, 3]. Works in chunks of a,
    because the full N1 x N2 table for two 10,000-point clouds would need 400 MB or more."""
    bx, by, bz = (b[:, None, :, d] for d in range(3))
    out = []
    for start in range(0, a.shape[1], chunk):
        part = a[:, start:start + chunk]
        dx, dy, dz = part[:, :, 0:1] - bx, part[:, :, 1:2] - by, part[:, :, 2:3] - bz
        out.append((dx * dx + dy * dy + dz * dz).argmin(dim=2))
    return torch.cat(out, dim=1)


def chamfer_distance(xyz1: torch.Tensor, xyz2: torch.Tensor, transpose: bool = False, sqrt: bool = False,
                     eps: float = 1e-12, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared distance from every point to its nearest point in the other cloud.

    xyz1 [B, N1, 3], xyz2 [B, N2, 3] -> dist1 [B, N1], dist2 [B, N2]. Square-rooted if sqrt is set.
    Gradients flow to both clouds through the nearest pairs, as in the CUDA backward kernel.
    chunk is not in the original: it only bounds memory."""
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1, xyz2 = xyz1.transpose(1, 2), xyz2.transpose(1, 2)
    if xyz1.shape[0] != xyz2.shape[0] or xyz1.shape[2] != 3 or xyz2.shape[2] != 3:
        raise ValueError(f"expected [B, N1, 3] and [B, N2, 3], got {tuple(xyz1.shape)} and {tuple(xyz2.shape)}")
    with torch.no_grad():
        idx1, idx2 = _nearest(xyz1, xyz2, chunk), _nearest(xyz2, xyz1, chunk)
    dist1 = (xyz1 - batch_index_select(xyz2, idx1, dim=1)).square().sum(-1)
    dist2 = (xyz2 - batch_index_select(xyz1, idx2, dim=1)).square().sum(-1)
    if sqrt:
        dist1, dist2 = torch.sqrt(torch.clamp(dist1, eps)), torch.sqrt(torch.clamp(dist2, eps))
    return dist1, dist2


def real_torkit3d_usable() -> bool:
    """True if the compiled torkit3d imports and there is a CUDA device for it to run on."""
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("torkit3d.ops.sample_farthest_points")
        return True
    except Exception:
        return False


def _stand_in_modules() -> dict:
    """A package that looks like torkit3d to `from torkit3d.x import y`, holding only what Point-SAM imports."""
    def module(name: str, **members) -> types.ModuleType:
        made = types.ModuleType(name)
        made.__dict__.update(members)
        made.__path__ = []  # marks it as a package, so submodule imports resolve
        made.is_fallback = True
        return made

    functional = module("torkit3d.nn.functional", batch_index_select=batch_index_select)
    fps = module("torkit3d.ops.sample_farthest_points", sample_farthest_points=sample_farthest_points)
    chamfer = module("torkit3d.ops.chamfer_distance", chamfer_distance=chamfer_distance)
    nn = module("torkit3d.nn", functional=functional)
    ops = module("torkit3d.ops", sample_farthest_points=fps, chamfer_distance=chamfer)
    return {"torkit3d": module("torkit3d", nn=nn, ops=ops), "torkit3d.nn": nn, "torkit3d.nn.functional": functional,
            "torkit3d.ops": ops, "torkit3d.ops.sample_farthest_points": fps, "torkit3d.ops.chamfer_distance": chamfer}


def install_torkit3d(force_fallback: bool = False) -> str:
    """Make `import torkit3d` work. Returns "torkit3d" if the real one will be used, "fallback" otherwise.

    Call this before anything imports pc_sam, because pc_sam binds the functions when it is first imported.
    To swap after that, use use_fallback()."""
    if not force_fallback and real_torkit3d_usable():
        return "torkit3d"
    if not getattr(sys.modules.get("torkit3d"), "is_fallback", False):
        for name in [name for name in sys.modules if name == "torkit3d" or name.startswith("torkit3d.")]:
            del sys.modules[name]
        sys.modules.update(_stand_in_modules())
    return "fallback"


@contextlib.contextmanager
def use_fallback():
    """Inside this block an already imported Point-SAM uses the fallback functions, even on CUDA.
    For comparing the two paths in one process (scripts/compare_fallback.py)."""
    common = importlib.import_module("pc_sam.model.common")
    saved = (common.sample_farthest_points, common.chamfer_distance, common.batch_index_select)
    common.sample_farthest_points, common.chamfer_distance = sample_farthest_points, chamfer_distance
    common.batch_index_select = batch_index_select
    try:
        yield
    finally:
        common.sample_farthest_points, common.chamfer_distance, common.batch_index_select = saved


def autocast(device) -> torch.autocast:
    """Mixed precision on CUDA, as every result so far was made. Full precision elsewhere: slower, but exact
    and free of the half-precision gaps in the Metal backend."""
    kind = torch.device(device).type
    return torch.autocast(kind if kind == "cuda" else "cpu", enabled=kind == "cuda")


def pick_device() -> str:
    """cuda if there is one, else Apple's mps, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
