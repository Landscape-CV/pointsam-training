#!/bin/bash
# Compare Point-SAM with the real torkit3d against the plain-PyTorch fallback, on one GPU: masks, speed, scores.
# See docs/no_cuda.md. The tables land in results/compare_fallback/ and at the end of the log.
#
# Submit from the repo root:
#   mkdir -p slurm_outs && sbatch scripts/slurm/compare_fallback.sbatch results/isprs_lora/best.safetensors
# Another GPU type, kept apart from the H100 run (sbatch flags go before the file, script flags after the checkpoint):
#   sbatch --gres=gpu:a40:1 -o slurm_outs/compare_fallback_a40_%j.out scripts/slurm/compare_fallback.sbatch \
#       results/isprs_lora/best.safetensors --out-dir results/compare_fallback_a40

#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J compare_fallback
#SBATCH -o slurm_outs/compare_fallback_%j.out

set -e
echo "job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

module load anaconda3
module load cuda/12.6.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${TREESEG_ENV:-$SLURM_SUBMIT_DIR/env}"

cd "$SLURM_SUBMIT_DIR"
CHECKPOINT="${1:-results/isprs_lora/best.safetensors}"
# anything after the checkpoint goes to the script, for example --out-dir results/compare_fallback_a40
python -u -W ignore scripts/compare_fallback.py --checkpoint "$CHECKPOINT" "${@:2}"
echo "done"
