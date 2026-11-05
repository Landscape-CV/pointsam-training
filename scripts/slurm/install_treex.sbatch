#!/bin/bash
# Add TreeX to the training environment. Installed packages other than pandas and scipy are held at their current versions,
# so if a dependency needs a different torch or numpy, this stops with an error.
# pointtree asks for psutil<7, but IPython here needs psutil>=7 and pointtree only uses psutil in its profiler,
# so pointtree goes in with --no-deps. TreeX gave the same clicks with psutil 7.2.2 on the Mac (30 Oct 2026).
#
# From the repo root:
#   sbatch scripts/slurm/install_treex.sbatch
# Success ends with TREEX_INSTALL_DONE

#SBATCH -p ice-cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J install_treex
#SBATCH -o slurm_outs/install/install_treex_%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
export PIP_CACHE_DIR="$SLURM_SUBMIT_DIR/../.cache/pip"
module load anaconda3/2023.03 cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SLURM_SUBMIT_DIR/env"

PINS="$(mktemp)"
python -m pip freeze | grep -vE '^-e| @ |^pandas==|^scipy==' > "$PINS"
# pointtorch needs pandas<3 and pygam needs scipy<1.17, so those two move. These are the versions TreeX and this repo's tests passed with on the Mac.
python -m pip install -c "$PINS" pandas==2.2.3 scipy==1.15.3 pygam==0.12.0 numba==0.68.1 pointtorch==1.1.0 \
    circle_detection==1.0.0 pyclesperanto-prototype rasterio scikit-image
# FORCE_CUDA builds the GPU code on a node with no GPU, the same way torkit3d is built in install_env.sbatch
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0" NVCC_APPEND_FLAGS="-std=c++20" \
    python -m pip install -c "$PINS" --no-build-isolation torch-scatter torch-cluster
python -m pip install --no-deps pointtree==1.0.1

python -c "from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS; import torch_scatter, torch_cluster; print('TreeX imports')"
python -c "import torch, numpy, psutil, IPython, torkit3d; print('torch', torch.__version__, 'numpy', numpy.__version__, 'psutil', psutil.__version__)"
python -m pytest tests -q -W ignore
echo TREEX_INSTALL_DONE
