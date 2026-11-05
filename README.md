# Point-SAM training for forest scans

Fine-tunes [Point-SAM](https://github.com/zyc00/Point-SAM) to separate individual trees in terrestrial LiDAR, then scores it on whole tiles. Part of the CS 6999 HAAG LiDAR project at Georgia Tech.

The training recipe, click methods and mask merging are adapted from Kevin Hu's work in `Landscape-CV/Ecomodel`, branch `jianwei/spcg`, and files that carry that code say so.

## Getting started

Training and scoring need an NVIDIA GPU, so they run on PACE-ICE. [docs/cluster.md](docs/cluster.md) has every command; in short:

1. **Clone** this repo into your ICE scratch folder.
2. **Install** the environment with `sbatch scripts/slurm/install_env.sbatch`. It ends in `INSTALL_DONE`.
3. **Get the data:** download `wytham.zip` from Zenodo and run `python scripts/prepare_data.py`.
4. **Add TreeX** with `sbatch scripts/slurm/install_treex.sbatch`, if you want TreeX clicks.
5. **Smoke test:** `bash scripts/slurm/submit.sh notebooks/02_train.ipynb configs/experiment/smoke.yaml`, then the full run with `isprs_lora.yaml`.

Only `01_auto_prompt.ipynb` runs without a GPU, for example on a Mac with the packages in `requirements.txt` plus TreeX.

## Data

The [ISPRS Wytham benchmark](https://doi.org/10.5281/zenodo.16875688) (Cherlet et al. 2026, CC BY 4.0), using the authors' three strips:

| Strip | Used for | Trees | Crops |
|---|---|---|---|
| train | Training | 431 | 182 |
| val | Picking the checkpoint and tuning | 178 | 56 |
| test | Final score, once | 245 | 56 |

Train and test share no trees. Val shares cut trees with both, so its numbers are never reported as results. Data isn't stored in git: `scripts/prepare_data.py` builds it from the original PLY files.

## Layout

```
src/treeseg/
  data/      prepare.py (voxels, tiles, crops), dataset.py, ground.py (ground removal and its checks)
  models/    point_sam.py, lora.py
  engine/    trainer.py, clicks.py, segment.py, scoring.py, benchmark.py
  utils/     curves.py, view3d.py
configs/     data and experiment settings
notebooks/   01_auto_prompt (no GPU), 02_train, 03_benchmark, 04_test_crops
scripts/     prepare_data.py, inspect_ground.py, slurm/
tests/       run with: python -m pytest tests
docs/        cluster.md (PACE-ICE setup)
```

## Clicks and scores

Point-SAM outlines whatever you click on, so every score depends on where the clicks came from.

- **Training and validation:** clicks are placed from the true labels (`val_click: stem` is 2 trunk clicks per tree). Validation numbers show training is working, and they are not a final score.
- **Test:** software picks the clicks, as in real use. `stems` is our stem finder, up to 2 clicks per tree. `treex` is the TreeX trunk detector, 1 click about 1.3 m up. `perfect` is 1 click per tree from the labels, an upper bound.
- `val_iou_pick`: IoU of the mask the model picks for itself. This is the number used to choose the checkpoint.
- `f1`: trees separated correctly, counting a match at IoU 0.5 or above. Missed trees and made-up trees both lower it.
- `mean_tree_iou`: each real tree's best overlap, missed trees counting as 0. `mean_matched_iou`: overlap for matched trees only.
- `ignore_non_tree` skips predictions that land mostly on non-tree points; `strict` counts them as wrong.

## Results so far

Measured on ICE, H100. Validation uses label clicks, so compare validation numbers only with each other.

| Run | Best `val_iou_pick` |
|---|---|
| `isprs_lora` (with augmentation) | 0.577 |
| `isprs_lora_noaug` | 0.561 |

Test strip, F1 at IoU 0.5, `ignore_non_tree`, mean of 3 tiles (`benchmark_clicks.yaml`):

| Model | `stems` | `treex` |
|---|---|---|
| Stock Point-SAM | 0.006 | 0.000 |
| Trained (`isprs_lora`) | 0.181 | 0.185 |

Training helps a lot. TreeX clicks reach more trees (48% against 28%) but F1 barely moves, because recall is about 0.12: the outlines, not the clicks, are the limit now. Tiles range from 0.10 to 0.31. The two click methods also use different numbers of clicks per tree. The test strip has now been used once, so tune on the val strip only.

## What a run writes

Nothing below is in git; copy anything you want to keep out of scratch.

- `results/<run>/`: `best.safetensors` (LoRA merged in, loads like stock Point-SAM), `last.safetensors`, `history.csv` (every logged step and validation) and `figures/`.
- `results/notebooks/`: the executed notebook for each run, with all outputs.
- `results/benchmark_clicks/`: `results.csv` (per tile and method), `summary.csv`, `f1_bars.png`.
- `slurm_outs/<what it tests>/`: job logs.

The `isprs_lora` checkpoints are in Nikhil's ICE scratch folder. A full training run takes about 20 minutes on an H100, so retraining is usually quicker than copying.

## Settings worth changing

All in `configs/experiment/<run>.yaml`, except crop size, which lives in `configs/data/isprs_wytham.yaml` and needs `prepare_data.py` rerun.

| Setting | Current | Notes |
|---|---|---|
| `lr`, `encoder_lr` | 1e-4, 2e-4 | `isprs_lora_lr5e5` and `isprs_lora_lr2e4` try others |
| `batch_size`, `grad_accum` | 1, 2 | `isprs_lora_batch4` tries 4 and 1 |
| `max_steps`, `val_every`, `early_stop_patience` | 3000, 200, 6 | 3000 steps is about 33 epochs |
| `num_points` | 10,000 | points the model sees per crop |
| `crop_m`, `stride_m` (data config) | 12 m, 6 m | |
| `lora_rank`, `lora_alpha` | 8, 16 | |
| `use_augmentation` | true | no clear difference so far |
| `click_mix` | stems 0.6, random 0.2, interior 0.2 | where training clicks go |

## Next

- Hyperparameter sweep on the val strip: the three configs above, then crop size and points per crop.
- Find why outlines fail: sort val-strip failures into spills onto neighbours, cut-short outlines and empty clicks.
- Fair click comparison: the same number of clicks per tree for `stems` and `treex`.
- Run the trained model on island scans, and label island trees in `instance-annotator`.

## Versions

`install_env.sbatch` checks out the Point-SAM commit our results were made with and installs packages at the versions in `requirements-lock.txt`, taken from the working ICE environment.

## Known limits

- One forest and one scanner. A good score here says nothing yet about other sites.
- Each crop gives the model 10,000 points over 12 m, so it can't learn fine branch detail.
- No colour: every point has the same grey feature.
- Crops have ground removed by the cloth filter, as in the benchmark, so understory is in training as non-tree.
- The model picks among its 3 masks using its own confidence score, which followed mask size in earlier tests on stock Point-SAM. `val_iou_first` next to `val_iou_pick` shows how much that costs.
