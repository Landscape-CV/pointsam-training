# Point-SAM training for forest scans

Fine-tunes [Point-SAM](https://github.com/zyc00/Point-SAM) to separate individual trees in terrestrial LiDAR, then scores it on whole tiles. Part of the CS 6999 HAAG LiDAR project at Georgia Tech.

The training recipe, click methods and mask merging are adapted from Kevin Hu's work in `Landscape-CV/Ecomodel`, branch `jianwei/spcg`, and files that carry that code say so.

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
notebooks/   01_auto_prompt (no GPU), 02_train, 03_benchmark
scripts/     prepare_data.py, inspect_ground.py, slurm/
tests/       run with: python -m pytest tests
docs/        cluster.md (PACE-ICE setup)
```

## Known limits

- One forest and one scanner. A good score here says nothing yet about other sites.
- Each crop gives the model 10,000 points over 12 m, so it can't learn fine branch detail.
- No colour: every point has the same grey feature.
- The model picks among its 3 masks using its own confidence score, which followed mask size in earlier tests on stock Point-SAM. `val_iou_first` next to `val_iou_pick` shows how much that costs.
