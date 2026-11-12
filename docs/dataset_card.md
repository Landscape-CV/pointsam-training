---
license: cc-by-4.0
task_categories:
  - other
tags:
  - point-cloud
  - lidar
  - forestry
  - tree-segmentation
  - 3d-segmentation
pretty_name: Wytham Woods ISPRS benchmark, original scan, tiles and crops
size_categories:
  - n<1K
---

# Wytham Woods (ISPRS benchmark): original scan, tiles and crops

The ISPRS Wytham benchmark is a terrestrial laser scan of temperate broadleaf forest in Wytham Woods, UK, in which every tree was segmented by hand. This repository holds an unchanged copy of the original scan (`raw/`), and the 30 m tiles and 12 m crops we cut from it for training and scoring a tree segmentation model.

This is not our data. It comes from Cherlet et al. (2026), https://doi.org/10.5281/zenodo.16875688, shared under CC BY 4.0. We re-share it under the same licence so every notebook in our code repository can run from one place. The authors' Zenodo record is the reference copy; if the two ever differ, theirs is right. If you use it, cite the original authors.

Code that made and uses these files: https://github.com/nisulay/pointsam-training. Its notebook `00_dataset.ipynb` downloads this dataset, or rebuilds the tiles and crops from `raw/`, and shows each step.

Model trained on it: https://huggingface.co/haag-lidar/point-sam-wytham-lora

## What we changed

Nothing in `raw/`. The tiles and crops were made from it like this:

- Thinned the scan to one point per 5 cm voxel.
- Kept the authors' three strips as the split: train, val and test.
- Cut each strip into 30 m tiles (`tiles/`).
- Cut 12 m crops with a 6 m step, removed ground with a cloth simulation filter, and kept up to 100,000 points per crop (`crops/`).
- Dropped colour. The scan has none.

The exact settings are in `summary.json`.

## The split

| Strip | Trees | Tiles | Crops |
|---|---|---|---|
| train | 431 | 9 | 182 |
| val | 178 | 3 | 56 |
| test | 245 (244 after thinning) | 3 | 56 |

- Train and test share no trees and sit 27 m apart.
- **Val shares some cut trees with train and with test**, because a tree on a strip boundary is cut in two. Use val to choose a checkpoint or to diagnose, not for a final score.
- **The test strip is for final scores only.** We have scored it once. Do not tune settings on it.

## Files

**`raw/`**, the authors' three files, unchanged, 3.3 GB in total:

| File | Size | Strip |
|---|---|---|
| `wytham_train.ply` | 2.05 GB | train |
| `wytham_val.ply` | 0.61 GB | val |
| `wytham_test.ply` | 0.60 GB | test |

**`tiles/`**, three files per tile, named `isprs_<strip>_t00_<number>`:

| File | Contents |
|---|---|
| `*_scan.laz` | The points, x y z in metres |
| `*_instances.npy` | One `int32` per point: a tree number above 0, or -1 for not a tree |
| `*_meta.json` | Tile name, strip, number of points and trees |

Ground is still in the tiles. The code removes it when a tile is loaded.

**`crops/train`, `crops/val`, `crops/test`**, one `.npz` per crop:

| Key | Contents |
|---|---|
| `xyz` | `float32`, up to 100,000 points by 3 |
| `instance_ids` | `int32`, same labels as above |

Ground is already removed from the crops.

## How to get it

```python
from huggingface_hub import snapshot_download

# tiles and crops only, about 0.9 GB: enough to train and score
snapshot_download("haag-lidar/wytham-isprs-tiles", repo_type="dataset", local_dir="data/wytham_isprs",
                  allow_patterns=["tiles/*", "crops/*", "summary.json"])
```

That puts `tiles/` and `crops/` where the configs in the code repository expect them. Leave out `allow_patterns` to get the original scan too, which is only needed to rebuild the tiles and crops from scratch.

## Limits

- One forest and one scanner. Nothing here says how a model trained on it does elsewhere.
- No colour or intensity.
- The same scan is also distributed with TreeLearn as `wytham_vox0.1.laz`. Do not mix that file into training if you test on this test strip, as the two overlap.

## Credit

Original data: Cherlet et al. (2026), ISPRS Wytham benchmark, CC BY 4.0, https://doi.org/10.5281/zenodo.16875688.

Processing: the CS 6999 HAAG LiDAR project at Georgia Tech.
