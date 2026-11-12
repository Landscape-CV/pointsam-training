---
license: apache-2.0
base_model: yuchen0187/Point-SAM
tags:
  - point-cloud
  - 3d-segmentation
  - lidar
  - forestry
  - tree-segmentation
---

# Point-SAM fine-tuned on Wytham Woods (terrestrial LiDAR)

[Point-SAM](https://github.com/zyc00/Point-SAM) (ViT-L) fine-tuned to outline individual trees in terrestrial laser scans. You give it a point on a tree, a "click", and it returns the points it thinks belong to that tree.

This is a research checkpoint from the CS 6999 HAAG LiDAR project at Georgia Tech. It is shared so teammates can reproduce and build on our results, and it is **not yet good enough to use as an automatic tree segmenter** (see Results).

Code, configs and notebooks: https://github.com/Landscape-CV/pointsam-training

## What changed from the original

The original weights are `yuchen0187/Point-SAM` (Apache 2.0). We fine-tuned them with LoRA adapters on the attention query and value projections (rank 8, alpha 16), and trained the patch embedding, prompt encoder, mask encoder and mask decoder. That is 5.8M of 311.8M parameters. The adapters are merged into the weights in this file, so it loads exactly like the original.

| Setting | Value |
|---|---|
| Training data | ISPRS Wytham benchmark, train strip: 431 trees, 182 crops of 12 m |
| Points per crop | 10,000, no colour (every point has the same feature) |
| Steps | 3,000, about 33 passes over the data |
| Learning rate | 1e-4, adapters 2e-4 |
| Checkpoint chosen by | `val_iou_pick` on the val strip (0.577) |
| Hardware | One NVIDIA H100, about 20 minutes |

Config: `configs/experiment/isprs_lora.yaml` at commit `18cf999`.

## Results

Whole 30 m tiles, F1 at IoU 0.5, one prediction matched to one tree. Predictions mostly on non-tree points are not counted against the model (`ignore_non_tree`).

**Test strip, 3 tiles, automatic clicks.** The test strip shares no trees with the training strip.

| Model | Stem-finder clicks | TreeX clicks |
|---|---|---|
| Original Point-SAM | 0.006 | 0.000 |
| This checkpoint | 0.181 | 0.185 |

**Val strip, 3 tiles**, where we looked at what limits the score. The val strip shares some cut trees with the training strip, so these are for diagnosis, not final results.

| Method | Trees with a click | Good mask before merging | Matched after merging | F1 |
|---|---|---|---|---|
| This checkpoint, TreeX clicks | 56% | 24% | 22% | 0.32 |
| This checkpoint, perfect clicks (from the labels) | 97% | 40% | 35% | 0.44 |
| TreeX alone, no model | 56% | n/a | 51% | 0.66 |

What this says:

- Training helps a great deal over the original weights.
- The masks are the limit. Even with a perfect click on every tree, only 40% of trees get a mask that overlaps the tree by half or more. The usual failure is the mask spilling onto a neighbouring tree.
- [TreeX](https://github.com/ai4trees/pointtree), a rule-based method with no training, scores higher on this forest than this checkpoint does.

On 12 m val crops with clicks placed from the labels, the model's own pick of its 3 candidate masks averages IoU 0.54, the best of the 3 averages 0.68, and after correction clicks it reaches 0.89.

## How to use

The model needs an NVIDIA GPU, because Point-SAM depends on a CUDA-only library (torkit3d). Set up the environment from the code repository, then:

```python
from huggingface_hub import hf_hub_download
from treeseg.models.point_sam import build_point_sam

path = hf_hub_download("haag-lidar/point-sam-wytham-lora", "model.safetensors")
model = build_point_sam(path, 512, 64).to("cuda").eval()
```

To segment a whole tile (ground removal, clicks, one mask per click, merging), see `notebooks/04_inference_steps.ipynb`, or point a config's `checkpoints:` entry at the downloaded file and run `notebooks/03_benchmark.ipynb`.

## Limits

- One forest (Wytham Woods, UK, temperate broadleaf) and one scanner. Nothing here says how it does elsewhere.
- Not tested with labels on any other site. On two tiles of a subtropical island scan, the TreeX clicks it relies on found only 2 or 3 trunks per tile, so the automatic pipeline does not carry over as is.
- Each crop gives the model 10,000 points over 12 m, so fine branch detail is not seen.
- No colour or intensity is used.
- All inference so far uses one positive click per tree. Negative clicks and correction rounds, which the model was trained with, are not yet used on whole tiles.

## Credits and licences

- Original model and weights: Zhou et al., *Point-SAM: Promptable 3D Segmentation Model for Point Clouds* (2024), arXiv:2406.17741. Weights `yuchen0187/Point-SAM`, Apache 2.0. This checkpoint is a modified version of those weights and is released under the same licence.
- Training data: the ISPRS Wytham benchmark (Cherlet et al. 2026), CC BY 4.0, https://doi.org/10.5281/zenodo.16875688.
- The training recipe, click methods and mask merging are adapted from Kevin Hu's work in `Landscape-CV/Ecomodel`.

```
@misc{zhou2024pointsampromptable3dsegmentation,
      title={Point-SAM: Promptable 3D Segmentation Model for Point Clouds},
      author={Yuchen Zhou and Jiayuan Gu and Tung Yen Chiang and Fanbo Xiang and Hao Su},
      year={2024},
      eprint={2406.17741},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.17741},
}
```
