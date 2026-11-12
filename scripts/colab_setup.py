"""
colab_setup.py - make a fresh Colab machine ready to run the notebooks.

Colab starts empty each session, so this fetches everything the notebooks read: the Python packages Colab lacks,
Point-SAM's code, our trained model and the tiles and crops. Finished steps are skipped, so rerunning is safe.
torkit3d is not compiled: without it, src/treeseg/models/torkit3d_fallback.py stands in, on the GPU too.

Run from the repo root, after cloning the repo on Colab (docs/colab.md has the cell to paste):
    python scripts/colab_setup.py            # enough for 00, 01 (without TreeX), 03 and 04
    python scripts/colab_setup.py --train    # also the stock Point-SAM weights that 02_train starts from
    python scripts/colab_setup.py --torkit3d # also compile the real torkit3d, for faster point sampling
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

POINT_SAM_COMMIT = "25f4fd9675479f3c951e830771595855ff7ed466"  # the commit every result so far was made with
PACKAGES = ["laspy[lazrs]", "plyfile", "cloth-simulation-filter", "timm", "safetensors", "hydra-core>=1.3", "huggingface_hub", "pyyaml"]
MODEL_REPO = "haag-lidar/point-sam-wytham-lora"
STOCK_WEIGHTS = "https://huggingface.co/yuchen0187/Point-SAM/resolve/main/model.safetensors"


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="also download the stock Point-SAM weights for 02_train")
    parser.add_argument("--torkit3d", action="store_true", help="compile the real torkit3d for faster point sampling")
    parser.add_argument("--skip-pip", action="store_true", help="leave the Python packages alone")
    parser.add_argument("--skip-data", action="store_true", help="do not download the tiles and crops")
    args = parser.parse_args()
    if not os.path.exists("src/treeseg"):
        sys.exit("run this from the repo root")

    #torch and torchvision are left as Colab ships them: they already match its GPU
    if not args.skip_pip:
        run(sys.executable, "-m", "pip", "install", "-q", *PACKAGES)

    if not os.path.isdir("thirdparty/Point-SAM"):
        run("git", "clone", "-q", "https://github.com/zyc00/Point-SAM.git", "thirdparty/Point-SAM")
    run("git", "-C", "thirdparty/Point-SAM", "checkout", "-q", POINT_SAM_COMMIT)

    #the compiled point sampling is about 30 times faster than the stand-in's Python loop. The build is the same as in
    #scripts/slurm/install_env.sbatch. If it fails, the stand-in is used and everything else still works
    if args.torkit3d:
        try:
            import torkit3d  # noqa: F401
            print("torkit3d is already built")
        except ImportError:
            try:
                run("git", "-C", "thirdparty/Point-SAM", "submodule", "update", "--init", "-q", "third_party/torkit3d")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "./thirdparty/Point-SAM/third_party/torkit3d"],
                               check=True, env={**os.environ, "FORCE_CUDA": "1", "NVCC_APPEND_FLAGS": "-std=c++20"})
                print("torkit3d built")
            except subprocess.CalledProcessError:
                print("TORKIT3D BUILD FAILED: carrying on with the plain-PyTorch stand-in, which is slower but gives the same masks")

    #the configs look for the trained model where a training run on ICE leaves it
    from huggingface_hub import hf_hub_download
    trained = "results/isprs_lora/best.safetensors"
    if not os.path.exists(trained):
        os.makedirs(os.path.dirname(trained), exist_ok=True)
        shutil.copy(hf_hub_download(MODEL_REPO, "model.safetensors"), trained)

    stock = "thirdparty/checkpoints/point_sam/model.safetensors"
    if args.train and not os.path.exists(stock):
        os.makedirs(os.path.dirname(stock), exist_ok=True)
        urllib.request.urlretrieve(STOCK_WEIGHTS, stock)

    sys.path.insert(0, "src")
    if not args.skip_data:
        from treeseg.data.download import fetch
        fetch("built", "data/wytham_isprs")

    #prove it works: load the model and outline one tree from one click
    import numpy as np
    import torch

    from treeseg.models.point_sam import build_point_sam
    from treeseg.models.torkit3d_fallback import pick_device
    from treeseg.models.torkit3d_fallback import real_torkit3d_usable
    device = pick_device()
    print("point sampling:", "compiled torkit3d" if real_torkit3d_usable() else "plain-PyTorch stand-in")
    print("python", sys.version.split()[0], "| torch", torch.__version__, "| device", device,
          torch.cuda.get_device_name(0) if device == "cuda" else "")
    model = build_point_sam(trained, 512, 64).to(device).eval()
    if os.path.exists("data/wytham_isprs/tiles/isprs_val_t00_00_scan.laz"):
        from treeseg.data.ground import load_tile
        from treeseg.engine.segment import click_candidates
        xyz, ids = load_tile("data/wytham_isprs/tiles", "isprs_val_t00_00")
        tree = xyz[ids == np.bincount(ids[ids > 0]).argmax()]
        click = np.median(tree[tree[:, 2] <= np.quantile(tree[:, 2], 0.1)], axis=0).reshape(1, 3)
        result = click_candidates(model, xyz, click, device=device)
        print("one click gave 3 masks of", result["masks"].sum(axis=1).tolist(), "points, scores", [round(float(s), 2) for s in result["scores"]])
    print("COLAB_SETUP_DONE")


if __name__ == "__main__":
    main()
