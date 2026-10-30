#!/bin/bash
# Run a notebook unattended on PACE-ICE. The finished copy, with outputs and curves, is saved under results/.
#
# Submit from the repo root:
#   mkdir -p slurm_outs results/notebooks
#   sbatch scripts/slurm/run_notebook.sbatch notebooks/02_train.ipynb configs/experiment/smoke.yaml
# Watch:  squeue -u $USER    then    tail -f slurm_outs/slurm_<jobid>.out
# Cancel: scancel <jobid>

#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J pointsam
#SBATCH -o slurm_outs/slurm_%j.out

set -e
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

module load anaconda3
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${TREESEG_ENV:-/storage/ice1/9/1/nsulay3/envs/pointsam}"

cd "$SLURM_SUBMIT_DIR"
NOTEBOOK="${1:-notebooks/02_train.ipynb}"
export TREESEG_CONFIG="${2:-configs/experiment/isprs_lora.yaml}"
RUN="$(basename "$TREESEG_CONFIG" .yaml)"
echo "notebook: $NOTEBOOK  config: $TREESEG_CONFIG"
jupyter nbconvert --to notebook --execute "$NOTEBOOK" \
    --ExecutePreprocessor.timeout=-1 \
    --output-dir results/notebooks --output "$(basename "$NOTEBOOK" .ipynb)_${RUN}.ipynb"
echo "done"
