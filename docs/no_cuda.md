# Running without an NVIDIA GPU

Point-SAM can run on a CPU or on an Apple Silicon Mac (PyTorch `mps`). Inference is practical. Training runs only with fewer target trees per crop than the recipe uses, and is far slower than ICE.

Everything below was measured on 11 Nov 2026 on a MacBook with an Apple M4 (10 cores, 24 GB), macOS 15.6, torch 2.6.0, with the public checkpoint `haag-lidar/point-sam-wytham-lora`.

## What blocked it, and the fix

| Blocker | Where | What was done |
|---|---|---|
| `torkit3d`, CUDA kernels only | Point-SAM imports 3 functions from it: `sample_farthest_points`, `batch_index_select`, `chamfer_distance` (`pc_sam/model/common.py`, `pc_sam.py`, `prompt_encoder.py`) | `src/treeseg/models/torkit3d_fallback.py` has plain-PyTorch versions. When the real one cannot run, a stand-in package named `torkit3d` is registered before Point-SAM is imported |
| `apex` fused LayerNorm | `replace_with_fused_layernorm` | Already handled: `build_point_sam` falls back to `nn.LayerNorm` |
| `torch.autocast("cuda")` and `GradScaler("cuda")` | our `engine/trainer.py` | Mixed precision is now on for CUDA only. Other devices use full precision |

No upstream Point-SAM file is edited. Other `.cuda()` calls in Point-SAM are in demo `main()` functions and in `pc_sam/model/build.py`, which this repo does not use.

The real `torkit3d` is used whenever CUDA is available and it imports, so nothing changes on ICE.

Farthest point sampling in the CUDA kernel always starts at index 0, and the fallback does the same. Of the three functions only two were CUDA code: `batch_index_select` was already plain PyTorch.

## Setup on a Mac

In an environment that has torch (here `pytlidar_v2`):

```bash
pip install torchvision==0.21.0 timm==1.0.30 safetensors hydra-core huggingface_hub
```

`torchvision` must match torch (0.21.0 goes with torch 2.6.0). Then clone Point-SAM at the pinned commit, as `scripts/slurm/install_env.sbatch` does, without building torkit3d:

```bash
git clone https://github.com/zyc00/Point-SAM.git thirdparty/Point-SAM
```

```bash
git -C thirdparty/Point-SAM checkout 25f4fd9675479f3c951e830771595855ff7ed466
```

## Inference

```python
from huggingface_hub import hf_hub_download
from treeseg.models.point_sam import build_point_sam
from treeseg.models.torkit3d_fallback import pick_device

device = pick_device()   # cuda, else mps, else cpu
path = hf_hub_download("haag-lidar/point-sam-wytham-lora", "model.safetensors")
model = build_point_sam(path, 512, 64).to(device).eval()
```

Pass `device=device` to `click_candidates`, `predict_click_masks`, `segment_tile` and `run_benchmark`. Their default is still `"cuda"`.

To time the devices on one machine:

```bash
python scripts/time_devices.py --checkpoint <model.safetensors> --whole-tile mps
```

Measured, tile `isprs_val_t00_00`, one click in a 12 m crop of 10,000 points:

| Device | Time per crop | Note |
|---|---|---|
| `cpu` | 0.54 s | median of 9 runs, range 0.54 to 0.55 |
| `mps` | 0.32 s | median of 9 runs, range 0.31 to 0.40 |

The same three crops on `cpu` and `mps` gave picked masks with IoU 1.000, 0.999 and 1.000 against each other.

A whole val tile on `mps` (36 to 77 clicks, about 3 million points) took 16 to 23 s for the model step. The ICE benchmark logged 12 to 26 s per tile for model, merging and scoring together, so the two are of the same order.

No operation was unsupported on `mps` during inference, and `PYTORCH_ENABLE_MPS_FALLBACK` was not set. Point sampling is run on the CPU even for `mps` inputs: its loop was about 6 times slower on Metal (0.14 s against 0.02 s for 10,000 points).

## Does it give the same answer as CUDA?

Yes, as far as it has been measured. On 11 Nov 2026 `scripts/compare_fallback.py` ran on ICE (job 6969973, NVIDIA A40, torch 2.14.1+cu126, checkpoint `results/isprs_lora/best.safetensors`). It runs the real torkit3d and the fallback on the same GPU, the same points and the same clicks.

**Masks.** On 10 single-click crops of `isprs_val_t00_00` the two paths gave identical masks: IoU 1.0000 for all 3 candidates of every crop, the same pick, the same 512 sampled points in the same order, and chamfer distances differing by 0.

**Scores.** Identical to 3 decimals, apart from one `raw_mean_tree_iou` (0.212 against 0.213 on one tile).

| Mean of 3 val tiles | F1 | Precision | Recall | Raw recall |
|---|---|---|---|---|
| `perfect` clicks, torkit3d | 0.444 | 0.609 | 0.349 | 0.402 |
| `perfect` clicks, fallback | 0.444 | 0.609 | 0.349 | 0.402 |
| `stems` clicks, torkit3d | 0.299 | 0.578 | 0.203 | 0.198 |
| `stems` clicks, fallback | 0.299 | 0.578 | 0.203 | 0.198 |

| 56 val crops, label clicks | val_loss | val_iou | val_iou_first | val_iou_pick |
|---|---|---|---|---|
| torkit3d | 0.2609 | 0.8858 | 0.6785 | 0.5436 |
| fallback | 0.2609 | 0.8858 | 0.6785 | 0.5436 |

**Speed on the A40.** The fallback is slower on a GPU, and nearly all of that is point sampling, which is a Python loop of 511 steps where torkit3d has one kernel.

| Median seconds | torkit3d | fallback | Ratio |
|---|---|---|---|
| One crop, one click | 0.155 | 0.245 | 1.58 |
| Point sampling alone (512 of 10,000) | 0.003 | 0.093 | 30 |
| Whole tile, mean | 13.1 | 15.9 | 1.21 |
| 56 val crops | 9.0 | 14.2 | 1.57 |

So on a CUDA machine keep the real torkit3d, which is what `build_point_sam` does by itself. The fallback is for machines where torkit3d cannot run.

The same job on an H100 (6969831, NVIDIA H100 80GB) gave the same result: identical masks on all 10 crops, the same tile scores as the table above, and val crops at 0.2615 / 0.8855 / 0.6786 / 0.5436 for both paths.

| Median seconds, H100 | torkit3d | fallback | Ratio |
|---|---|---|---|
| One crop, one click | 0.122 | 0.170 | 1.40 |
| Point sampling alone (512 of 10,000) | 0.0015 | 0.049 | 32 |
| Whole tile, mean | 10.8 | 12.3 | 1.13 |
| 56 val crops | 6.3 | 8.0 | 1.28 |

Limits of this check: two GPU types, one checkpoint, inference and validation only. Training with the fallback on CUDA was not compared.

To repeat it, on ICE from the repo root:

```bash
sbatch scripts/slurm/compare_fallback.sbatch results/isprs_lora/best.safetensors
```

The tables print at the end of `slurm_outs/compare_fallback_<jobid>.out` and are saved in `results/compare_fallback/`.

Across machines, the Mac on `mps` with the fallback scored the 56 val crops at val_loss 0.261, val_iou 0.887, val_iou_first 0.679, val_iou_pick 0.544, which matches the table above. Whole-tile F1 on the Mac was 0.430 (`perfect`) and 0.302 (`stems`), with single tiles up to 0.065 away from ICE. Since the fallback is now shown to be exact, that gap comes from something else that differs between the machines. It is not the checkpoint: the Hugging Face file and `results/isprs_lora/best.safetensors` on ICE have the same SHA-256. Ground removal is known to differ between machines and changes both the points and the clicks; this was not isolated further.

## More checks on the Mac

All on 11 Nov 2026, with the public checkpoint unless stated.

| Check | Result |
|---|---|
| `cpu` against `mps`, all 77 label clicks of `isprs_val_t00_00` | picked-mask IoU mean 0.9999, lowest 0.9972; same pick of the 3 candidates every time; scores differ by at most 0.0013 |
| `mps` twice | identical masks and scores |
| One training gradient, `cpu` against `mps`, same crop and clicks | loss 2.65958 against 2.65963; cosine 0.99999996 over all 243 trainable tensors; relative difference 0.03%; no NaN |
| 100 training steps on `mps` from the stock weights, `num_masks: 2`, 8 val crops | val_loss 3.16 to 0.77, val_iou 0.12 to 0.66, val_iou_pick 0.066 to 0.166; 10.7 min including two validations |
| Saved `best.safetensors` and `last.safetensors` reloaded | the same 4 val scores as the model in memory, to 4 decimals |
| `run_benchmark(cfg, device="mps")`, 3 val tiles, `perfect` and `stems` clicks | completes in 127 s; saved masks and `keep` replay to exactly the saved prediction |

The 100-step run shows that training on `mps` learns and saves correctly. It says nothing about the quality of a full run: on ICE the same recipe with 8 target trees reached val_iou_pick 0.345 by step 200.

## Training

It runs on `mps` in full precision, but not with the recipe's 8 target trees per crop on a 24 GB Mac. Measured with `isprs_lora.yaml`, changing only `num_masks`, starting from the fine-tuned checkpoint (the stock one was not downloaded; the cost per step is the same):

| `num_masks` | Time per optimizer step (2 crops) | Metal memory after the forward pass | Outcome |
|---|---|---|---|
| 2 | 3.5 to 4.5 s | 6.9 GB tensors, 8.3 GB held by the driver | runs |
| 4 | 5.9 s | 10.2 GB tensors, 12.8 GB driver | runs |
| 8 (the recipe) | 100 to 300 s | 18.7 GB tensors, 21.8 GB driver | runs, but the Mac swaps to disk. Stopped after 3 steps |

On ICE a full 3,000-step run takes about 18 minutes, 0.36 s per step. With `num_masks: 4` the Mac would need about 5 hours for 3,000 steps, and each step would see half as many target trees. Raising `grad_accum` to 4 restores the count and doubles the time. Neither changed recipe has been trained to the end, so its quality is unknown.

Half precision does not work for training on `mps` in torch 2.6: with `torch.autocast("mps", dtype=torch.float16)` the forward pass ran (8.8 GB at 4 masks) and the backward pass stopped with `RuntimeError: Expected scalar_type == ScalarType::Float ...`.

Training on `cpu` was not tried beyond one forward and backward pass for the gradient check above, which took 3.9 s with `num_masks: 2` (2.8 s on `mps`).

## Tests

`tests/test_torkit3d_fallback.py` checks each function against a simple reference: point sampling against a one-point-at-a-time loop (also at the real size, 512 of 10,000 points, and with repeated points), index select against `torch.gather`, chamfer against a full distance matrix, and its gradient against the CUDA backward kernel's formula. One test runs Point-SAM's own grouping and click sampling on the fallback, and is skipped when `thirdparty/Point-SAM` is not cloned. They run on `cpu`, and on `mps` when present: 26 tests, 81 with the rest of the suite.
