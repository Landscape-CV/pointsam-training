# Running on Google Colab

Colab gives a GPU with no queue. From VS Code, the Colab extension runs a notebook's cells on a Colab machine while the notebook file stays on your computer. That machine starts empty each session and cannot see your disk, so the first cell has to fetch the code, the model and the data.

**Status: the setup script was tested in a fresh clone on a Mac (12 Nov 2026), not yet on Colab.** Colab runs a newer Python than the 3.11 our results were made with.

## Steps

1. Open a notebook from this repo in VS Code, choose **Select Kernel**, then **Colab**, and pick a GPU runtime.
2. Add this as the first cell and run it. It takes a few minutes the first time in a session.

```python
import os, subprocess
REPO = "/content/pointsam-training"
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "-b", "mask-experiments", "https://github.com/Landscape-CV/pointsam-training.git", REPO], check=True)
os.chdir(REPO)
!python scripts/colab_setup.py
```

It ends with `COLAB_SETUP_DONE`, after printing the GPU it found and the result of one test click. Change `mask-experiments` to the branch you want, such as `main`. Add `--train` to the last line before running `02_train`, to also fetch the stock Point-SAM weights.

3. Run the notebook's own cells as usual. Don't commit the added cell.

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
