# Running on PACE-ICE

Connect the GT VPN first, then `ssh nsulay3@login-ice.pace.gatech.edu`. Home is small (30 GB), so everything goes in scratch: `/storage/ice1/9/1/nsulay3`. Scratch can be purged, so copy results you want to keep back to your Mac.

## One-time setup (login node)

```bash
SCRATCH=/storage/ice1/9/1/nsulay3
cd $SCRATCH/pointsam-training   # copied from your Mac, see "Copy code and data" below

# Point-SAM and its one compiled dependency
mkdir -p thirdparty/checkpoints/point_sam
git clone https://github.com/zyc00/Point-SAM.git thirdparty/Point-SAM
git -C thirdparty/Point-SAM submodule update --init third_party/torkit3d

# Python environment in scratch
module load anaconda3
conda create --prefix $SCRATCH/envs/pointsam python=3.11 -y
conda activate $SCRATCH/envs/pointsam
module load cuda/12.6.1
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# torkit3d needs a GPU compile. FORCE_CUDA lets it build on a login node with no GPU
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0" pip install ./thirdparty/Point-SAM/third_party/torkit3d

# Model weights
python -c "from huggingface_hub import hf_hub_download; import shutil; shutil.copy(hf_hub_download('yuchen0187/Point-SAM', 'model.safetensors'), 'thirdparty/checkpoints/point_sam/model.safetensors')"
```

- If torkit3d fails inside PyTorch's own headers, add `NVCC_APPEND_FLAGS="-std=c++20"` before `pip install`.
- apex is not needed for this code.

## Copy code and data (do this first, on your Mac, VPN on)

The configs look for the data one level above the code, at `../Dataset/pointsam_isprs`. Copying the code avoids setting up GitHub login on ICE for a private repo.

```bash
ssh nsulay3@login-ice.pace.gatech.edu "mkdir -p /storage/ice1/9/1/nsulay3/Dataset"
rsync -avh --progress --exclude .git --exclude results --exclude thirdparty /Users/nikhilclaude/Documents/Projects/lidar_project/pointsam-training nsulay3@login-ice.pace.gatech.edu:/storage/ice1/9/1/nsulay3/
rsync -avh --progress --exclude inspect /Users/nikhilclaude/Documents/Projects/lidar_project/Dataset/pointsam_isprs nsulay3@login-ice.pace.gatech.edu:/storage/ice1/9/1/nsulay3/Dataset/
```

## Run a notebook

```bash
mkdir -p slurm_outs results/notebooks
sbatch scripts/slurm/run_notebook.sbatch notebooks/02_train.ipynb configs/experiment/smoke.yaml
```

Watch it with `squeue -u $USER` and `tail -f slurm_outs/slurm_<jobid>.out`, and cancel with `scancel <jobid>`. The finished notebook is saved to `results/notebooks/`.

- **Order:** `smoke.yaml` first, then `isprs_lora.yaml`, then `03_benchmark.ipynb` with `configs/experiment/benchmark.yaml`.
- **GPU type:** to see what's free, run `sinfo -p ice-gpu -o "%G %t %D"`. To use another NVIDIA GPU, change `--gres` in the sbatch file. AMD GPUs won't work.
- **Queue limit:** `ice-gpu` jobs can run for up to 16 hours.
