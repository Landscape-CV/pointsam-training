"""The plain-PyTorch stand-ins for torkit3d, each checked against the simplest thing that could be right."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from treeseg.models import torkit3d_fallback as fb

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def naive_fps(points: torch.Tensor, k: int) -> list[int]:
    """The CUDA kernel's loop written out one point at a time: start at 0, strict > so ties keep the lowest index."""
    points = points.tolist()
    min_dist = [float("inf")] * len(points)
    selected, picks = 0, [0]
    for _ in range(1, k):
        px, py, pz = points[selected]
        max_dist, max_idx = 0.0, selected
        for j, (x, y, z) in enumerate(points):
            #float32 arithmetic, as in the kernel
            d = float(torch.tensor((x - px), dtype=torch.float32) ** 2 + torch.tensor((y - py), dtype=torch.float32) ** 2
                      + torch.tensor((z - pz), dtype=torch.float32) ** 2)
            min_dist[j] = min(min_dist[j], d)
            if min_dist[j] > max_dist:
                max_dist, max_idx = min_dist[j], j
        selected = max_idx
        picks.append(selected)
    return picks


@pytest.mark.parametrize("device", DEVICES)
def test_fps_matches_naive_loop(device):
    points = torch.rand(3, 60, 3, generator=torch.Generator().manual_seed(0))
    got = fb.sample_farthest_points(points.to(device), 12)
    assert got.shape == (3, 12) and got.dtype == torch.long and got.device.type == device
    for b in range(3):
        assert got[b].tolist() == naive_fps(points[b], 12)


def test_fps_starts_at_index_zero_and_never_repeats():
    points = torch.rand(2, 200, 3, generator=torch.Generator().manual_seed(1))
    got = fb.sample_farthest_points(points, 50)
    assert (got[:, 0] == 0).all()
    assert all(len(set(row.tolist())) == 50 for row in got)


def test_fps_known_answer_on_a_line():
    #from 0 the furthest is 10, then 5. Points 2, 3, 7 and 8 are then all 2 away from a pick, and the lowest index wins
    points = torch.tensor([[[float(i), 0.0, 0.0] for i in range(11)]])
    assert fb.sample_farthest_points(points, 4).tolist() == [[0, 10, 5, 2]]


def test_fps_transpose_and_all_points_equal():
    points = torch.rand(2, 40, 3, generator=torch.Generator().manual_seed(2))
    assert torch.equal(fb.sample_farthest_points(points.transpose(1, 2), 9, transpose=True), fb.sample_farthest_points(points, 9))
    #nothing is further than 0 away, so the kernel keeps repeating its last pick, index 0
    assert fb.sample_farthest_points(torch.ones(1, 8, 3), 4).tolist() == [[0, 0, 0, 0]]


def test_fps_rejects_bad_sizes():
    with pytest.raises(ValueError):
        fb.sample_farthest_points(torch.rand(1, 5, 3), 6)
    with pytest.raises(ValueError):
        fb.sample_farthest_points(torch.rand(1, 5, 2), 2)


@pytest.mark.parametrize("device", DEVICES)
def test_batch_index_select_matches_gather(device):
    generator = torch.Generator().manual_seed(3)
    values = torch.rand(4, 30, 5, generator=generator).to(device)
    index = torch.randint(0, 30, (4, 7), generator=generator).to(device)
    got = fb.batch_index_select(values, index, dim=1)
    assert got.shape == (4, 7, 5)
    assert torch.equal(got, torch.gather(values, 1, index.unsqueeze(-1).expand(-1, -1, 5)))
    assert torch.equal(got, torch.stack([values[b, index[b]] for b in range(4)]))


def test_batch_index_select_one_index_per_item_drops_the_dimension():
    #how Point-SAM picks one of its 3 candidate masks: [B, 3, N] with index [B] gives [B, N]
    masks = torch.rand(6, 3, 20, generator=torch.Generator().manual_seed(4))
    chosen = torch.tensor([0, 2, 1, 1, 0, 2])
    got = fb.batch_index_select(masks, chosen, dim=1)
    assert got.shape == (6, 20)
    assert torch.equal(got, masks[torch.arange(6), chosen])


def test_batch_index_select_other_dim():
    values = torch.rand(2, 4, 9, generator=torch.Generator().manual_seed(5))
    index = torch.tensor([[0, 8, 3], [5, 5, 1]])
    assert torch.equal(fb.batch_index_select(values, index, dim=2), torch.gather(values, 2, index[:, None, :].expand(-1, 4, -1)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [1024, 7])
def test_chamfer_matches_distance_matrix(device, chunk):
    generator = torch.Generator().manual_seed(6)
    a, b = torch.rand(2, 50, 3, generator=generator), torch.rand(2, 33, 3, generator=generator)
    matrix = (a[:, :, None, :] - b[:, None, :, :]).square().sum(-1)  # [B, 50, 33], squared distances
    dist1, dist2 = fb.chamfer_distance(a.to(device), b.to(device), chunk=chunk)
    assert dist1.shape == (2, 50) and dist2.shape == (2, 33)
    assert torch.allclose(dist1.cpu(), matrix.min(dim=2).values, atol=1e-6)
    assert torch.allclose(dist2.cpu(), matrix.min(dim=1).values, atol=1e-6)


def test_chamfer_sqrt_unbatched_and_transpose():
    generator = torch.Generator().manual_seed(7)
    a, b = torch.rand(20, 3, generator=generator), torch.rand(15, 3, generator=generator)
    plain1, plain2 = fb.chamfer_distance(a, b)
    assert plain1.shape == (1, 20) and plain2.shape == (1, 15)
    root1, _ = fb.chamfer_distance(a, b, sqrt=True)
    assert torch.allclose(root1, torch.cdist(a, b).min(dim=1).values[None], atol=1e-6)
    flipped1, flipped2 = fb.chamfer_distance(a[None].transpose(1, 2), b[None].transpose(1, 2), transpose=True)
    assert torch.equal(flipped1, plain1) and torch.equal(flipped2, plain2)


def test_chamfer_gradient_matches_the_cuda_backward_kernel():
    #the kernel sends 2 * (p - nearest) to p and the opposite to its nearest point
    generator = torch.Generator().manual_seed(8)
    a = torch.rand(1, 12, 3, generator=generator, requires_grad=True)
    b = torch.rand(1, 9, 3, generator=generator, requires_grad=True)
    dist1, dist2 = fb.chamfer_distance(a, b)
    (dist1.sum() + dist2.sum()).backward()
    matrix = (a.detach()[:, :, None, :] - b.detach()[:, None, :, :]).square().sum(-1)
    idx1, idx2 = matrix.argmin(dim=2)[0], matrix.argmin(dim=1)[0]
    grad_a, grad_b = torch.zeros(12, 3), torch.zeros(9, 3)
    for i in range(12):
        g = 2 * (a.detach()[0, i] - b.detach()[0, idx1[i]])
        grad_a[i] += g
        grad_b[idx1[i]] -= g
    for j in range(9):
        g = 2 * (b.detach()[0, j] - a.detach()[0, idx2[j]])
        grad_b[j] += g
        grad_a[idx2[j]] -= g
    assert torch.allclose(a.grad[0], grad_a, atol=1e-6) and torch.allclose(b.grad[0], grad_b, atol=1e-6)


def test_install_gives_point_sam_its_three_imports(monkeypatch):
    for name in [name for name in sys.modules if name == "torkit3d" or name.startswith("torkit3d.")]:
        monkeypatch.delitem(sys.modules, name)
    assert fb.install_torkit3d(force_fallback=True) == "fallback"
    #the exact import lines in pc_sam/model/common.py
    from torkit3d.nn.functional import batch_index_select
    from torkit3d.ops.chamfer_distance import chamfer_distance
    from torkit3d.ops.sample_farthest_points import sample_farthest_points

    assert (batch_index_select, chamfer_distance, sample_farthest_points) == (fb.batch_index_select, fb.chamfer_distance, fb.sample_farthest_points)
    assert fb.install_torkit3d(force_fallback=True) == "fallback"  # a second call changes nothing
    for name in [name for name in sys.modules if name == "torkit3d" or name.startswith("torkit3d.")]:
        monkeypatch.delitem(sys.modules, name)


def test_real_torkit3d_is_not_used_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert fb.real_torkit3d_usable() is False


def test_autocast_is_on_only_for_cuda():
    with fb.autocast("cpu"):
        assert not torch.is_autocast_enabled("cpu")
    with fb.autocast("mps"):
        assert not torch.is_autocast_enabled("cpu")
    #without a GPU torch switches a cuda autocast off by itself, so only the request can be checked here
    assert fb.autocast("cuda:0").device == "cuda"
    assert fb.autocast("mps").device == "cpu"


def numpy_fps(points, k: int) -> list[int]:
    """The same loop as naive_fps, one pick at a time in numpy float32, fast enough for a full-size crop."""
    import numpy as np

    points = points.numpy().astype(np.float32)
    min_dist = np.full(len(points), np.inf, dtype=np.float32)
    selected, picks = 0, [0]
    for _ in range(1, k):
        d = points - points[selected]
        min_dist = np.minimum(min_dist, d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])
        candidate = int(min_dist.argmax())  # first of any ties, as strict > gives
        selected = candidate if min_dist[candidate] > 0 else selected
        picks.append(selected)
    return picks


@pytest.mark.parametrize("device", DEVICES)
def test_fps_at_the_size_the_model_uses(device):
    #10,000 points and 512 picks, the numbers build_point_sam is given
    points = torch.rand(1, 10000, 3, generator=torch.Generator().manual_seed(9)) * 2 - 1
    got = fb.sample_farthest_points(points.to(device), 512)[0].tolist()
    assert got == numpy_fps(points[0], 512)
    assert len(set(got)) == 512


def test_fps_with_repeated_points_picks_each_place_once():
    #the dataset pads a small crop by repeating points, so two indices can sit at the same place
    base = torch.rand(300, 3, generator=torch.Generator().manual_seed(10))
    points = torch.cat([base, base[:200]])[None]  # 500 points, 300 different places
    got = fb.sample_farthest_points(points, 300)[0]
    places = {tuple(row) for row in points[0, got].tolist()}
    assert len(places) == 300
    assert got.tolist() == numpy_fps(points[0], 300)


@pytest.mark.parametrize("device", DEVICES)
def test_chamfer_at_the_size_training_uses(device):
    #a mask-shaped split of a 10,000-point crop, as the click sampler calls it
    points = torch.rand(10000, 3, generator=torch.Generator().manual_seed(11)) * 2 - 1
    inside = points[:, 0] < -0.2
    a, b = points[inside][None], points[~inside][None]
    dist1, dist2 = fb.chamfer_distance(a.to(device), b.to(device))
    exact = torch.cdist(a.double(), b.double()).square()
    assert torch.allclose(dist1.cpu().double(), exact.min(dim=2).values, atol=1e-6)
    assert torch.allclose(dist2.cpu().double(), exact.min(dim=1).values, atol=1e-6)


POINT_SAM = os.path.join(os.path.dirname(__file__), "..", "thirdparty", "Point-SAM")
needs_point_sam = pytest.mark.skipif(not os.path.isdir(os.path.join(POINT_SAM, "pc_sam")), reason="thirdparty/Point-SAM is not cloned")


@needs_point_sam
@pytest.mark.parametrize("device", DEVICES)
def test_point_sam_grouping_runs_on_this_machine(device):
    """Point-SAM's own first step (sample centres, gather neighbours) with whatever torkit3d this machine gets."""
    from treeseg.models.point_sam import add_point_sam_to_path

    add_point_sam_to_path(POINT_SAM)
    from pc_sam.model.common import KNNGrouper, sample_fixed_points

    generator = torch.Generator().manual_seed(12)
    xyz = (torch.rand(2, 2000, 3, generator=generator) * 2 - 1).to(device)
    out = KNNGrouper(64, 16)(xyz, torch.zeros_like(xyz))
    assert out["centers"].shape == (2, 64, 3) and out["knn_idx"].shape == (2, 64, 16) and out["features"].shape == (2, 64, 16, 6)
    assert (out["fps_idx"][:, 0] == 0).all()
    #each centre is its own nearest neighbour
    nearest = torch.gather(xyz, 1, out["knn_idx"].flatten(1).unsqueeze(-1).expand(-1, -1, 3)).reshape(2, 64, 16, 3)
    assert ((nearest - out["centers"][:, :, None]).norm(dim=-1).min(dim=2).values < 1e-6).all()

    #the training click sampler, which is where chamfer_distance is used: one click per mask, inside the mask
    masks = (xyz[:, None, :, 0] < torch.tensor([-0.3, 0.4], device=device)[None, :, None])  # [2, 2, 2000]
    clicks, labels = sample_fixed_points(xyz, masks, None, from_error_region=True)
    assert clicks.shape == (4, 1, 3) and labels.shape == (4, 1) and labels.all()
    assert (clicks[:2, 0, 0] < 0.4).all()
