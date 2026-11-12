# Running on Google Colab

Colab gives a GPU with no queue. From VS Code, the Colab extension runs a notebook's cells on a Colab machine while the notebook file stays on your computer. That machine starts empty each session and cannot see your disk, so the first cell has to fetch the code, the model and the data.

**Status: the setup script was tested in a fresh clone on a Mac (12 Nov 2026), not yet on Colab.** Colab runs a newer Python than the 3.11 our results were made with.

## Steps

1. Open a notebook from this repo in VS Code, choose **Select Kernel**, then **Colab**, and pick a GPU runtime. Reuse the same runtime for the next notebook, so the files one notebook makes are there for the next.
2. Run the notebook from the top. Its first code cell notices it is on Colab, clones this repo to `/content/pointsam-training` and runs `scripts/colab_setup.py`. That takes a few minutes the first time in a session and ends with `COLAB_SETUP_DONE`, after printing the GPU it found and the result of one test click. On a Mac or on ICE the cell does nothing.
3. To use a config other than the notebook's default, uncomment the last line of that first cell and name the config there.

The first cell clones the branch named in its `BRANCH` line. `04_inference_steps` does not have the cell yet; paste the first code cell of `03_benchmark` at its top.

To train only on whole trees and score the result:

| Notebook | Config to name in the first cell | Makes |
|---|---|---|
| `02_train` | `configs/experiment/isprs_lora_whole.yaml` | `results/isprs_lora_whole/best.safetensors` |
| `03_benchmark` | `configs/experiment/whole_targets_val.yaml` | `results/whole_targets_val/summary.csv` |

## What it sets up

| What | From | Where it lands |
|---|---|---|
| This repo | GitHub | `/content/pointsam-training` |
| Packages Colab lacks | pip | Colab's Python. torch is left as Colab ships it |
| Point-SAM code, pinned commit | GitHub | `thirdparty/Point-SAM` |
| Our trained model | `haag-lidar/point-sam-wytham-lora` | `results/isprs_lora/best.safetensors` |
| Tiles and crops (0.9 GB) | `haag-lidar/wytham-isprs-tiles` | `data/wytham_isprs` |

## Limits

- **TreeX is not installed**, so use `perfect` or `stems` clicks. After the notebook's first cell, set `cfg["click_methods"] = ["perfect"]` if the config lists `treex`.
- **torkit3d is not compiled.** The plain-PyTorch stand-in runs on the GPU and gave identical masks on ICE, about 1.6 times slower per crop (see [no_cuda.md](no_cuda.md)).
- **Results live on the Colab machine** and are lost when the session ends. Download anything you want to keep.
- **Only pushed code is used.** Changes on your computer that aren't on GitHub don't reach Colab, apart from the notebook's own cells.
