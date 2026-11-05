#!/bin/bash
# One-time setup on PACE-ICE: Point-SAM, the Python environment in env/, torkit3d and the model weights.
# This runs as a job because login nodes kill heavy work. Rerunning it is safe: finished steps are skipped.
#
# Submit from the repo root:
#   mkdir -p slurm_outs && sbatch scripts/slurm/install_env.sbatch

#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J pointsam_install
#SBATCH -o slurm_outs/install_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ROOT="$(pwd)"
mkdir -p thirdparty/checkpoints/point_sam slurm_outs results/notebooks

# caches stay in scratch, away from the small home quota
export PIP_CACHE_DIR="$ROOT/../.cache/pip" CONDA_PKGS_DIRS="$ROOT/../.conda_pkgs"

module load anaconda3/2023.03 cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"

# the Point-SAM commit every result so far was made with
POINT_SAM_COMMIT=25f4fd9675479f3c951e830771595855ff7ed466
[ -d thirdparty/Point-SAM ] || git clone -q https://github.com/zyc00/Point-SAM.git thirdparty/Point-SAM
git -C thirdparty/Point-SAM checkout -q "$POINT_SAM_COMMIT"
git -C thirdparty/Point-SAM submodule update --init -q third_party/torkit3d
echo "Point-SAM commit: $(git -C thirdparty/Point-SAM rev-parse --short HEAD)"

[ -d env ] || conda create -q --prefix "$ROOT/env" python=3.11 -y > /dev/null
conda activate "$ROOT/env"
# torch and torchvision must come from the same CUDA build. timm needs torchvision, and installing it from
# the default index gives a mismatched build that fails on import ("operator torchvision::nms does not exist")
# requirements-lock.txt holds every package at the version our results were made with. PyPI is added because
# some locked dependencies of torch were later upgraded from there and the CUDA index may not carry them.
pip install -q -c requirements-lock.txt torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple
pip install -q -c requirements-lock.txt -r requirements.txt

# --no-build-isolation lets the build see the torch installed above. C++20 is for newer PyTorch headers.
# FORCE_CUDA builds the GPU code on a node with no GPU; the arch list covers ICE's NVIDIA generations.
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0" NVCC_APPEND_FLAGS="-std=c++20" \
    pip install -q --no-build-isolation ./thirdparty/Point-SAM/third_party/torkit3d

WEIGHTS=thirdparty/checkpoints/point_sam/model.safetensors
[ -s "$WEIGHTS" ] || curl -fL --retry 5 -C - -o "$WEIGHTS" https://huggingface.co/yuchen0187/Point-SAM/resolve/main/model.safetensors

python - <<'PY'
import os

import CSF
import hydra
import timm
import torch
import torchvision
import torkit3d

print("torch", torch.__version__, "torchvision", torchvision.__version__, "CUDA build", torch.version.cuda)
print("weights MB", os.path.getsize("thirdparty/checkpoints/point_sam/model.safetensors") // 10**6)
# a fresh clone has no data yet; scripts/prepare_data.py builds it with this environment
crops = "data/wytham_isprs/crops/train"
print("train crops", len(os.listdir(crops)) if os.path.isdir(crops) else "none yet")
PY
python -m pytest tests -q -W ignore
echo INSTALL_DONE
