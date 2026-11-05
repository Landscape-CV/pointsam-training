# Running on PACE-ICE

Connect the GT VPN first, then `ssh nsulay3@login-ice.pace.gatech.edu`. Home is small (30 GB), so everything goes in scratch: `/storage/ice1/9/1/nsulay3`. Scratch can be purged, so copy results you want to keep back to your Mac.

## One-time setup

Copy the code and data first (next section). Then, from the project folder on ICE:

```bash
mkdir -p slurm_outs && sbatch scripts/slurm/install_env.sbatch
```

This runs as a job because login nodes kill heavy work. It downloads Point-SAM and its weights, builds `env/`, compiles torkit3d and runs the tests. The log is `slurm_outs/install_<jobid>.out`, ending in `INSTALL_DONE`. Rerunning it skips finished steps.

## Copy code and data (do this first, on your Mac, VPN on)

Data lives inside the project at `data/wytham_isprs`, and git ignores it. Copying the code avoids setting up GitHub login on ICE for a private repo.

**Never add `--delete` or `--delete-excluded`.** They remove anything on ICE that this copy leaves out, including `env/`, `thirdparty/` and `results/`.

```bash
rsync -avh --progress --exclude .git --exclude env --exclude results --exclude thirdparty --exclude slurm_outs --exclude __pycache__ --exclude .pytest_cache --exclude inspect /Users/nikhilclaude/Documents/Projects/lidar_project/pointsam-training nsulay3@login-ice.pace.gatech.edu:/storage/ice1/9/1/nsulay3/
```

## Run a notebook

```bash
mkdir -p slurm_outs results/notebooks
sbatch scripts/slurm/run_notebook.sbatch notebooks/02_train.ipynb configs/experiment/smoke.yaml
```

Check it with `squeue -u $USER`, and cancel with `scancel <jobid>`. Watch training live with `tail -F slurm_outs/slurm_<jobid>.out`. The finished notebook is saved to `results/notebooks/`.

For runs you want to find later, use `bash scripts/slurm/submit.sh <notebook> <config>` instead. It gives the job a readable name and writes its log to a folder for what the run is testing, such as `slurm_outs/augmentation_test/no_augmentation_<jobid>.out`. Slurm does not create that folder itself, so the script does.

- **Order:** `smoke.yaml` first, then `isprs_lora.yaml`, then `03_benchmark.ipynb` with `configs/experiment/benchmark.yaml`.
- **GPU type:** to see what's free, run `sinfo -p ice-gpu -o "%G %t %D"`. To use another NVIDIA GPU, change `--gres` in the sbatch file. AMD GPUs won't work.
- **Queue limit:** `ice-gpu` jobs can run for up to 16 hours.

## TreeX for clicks (optional, needed by `click_methods: treex`)

With the environment active, on a compute node because login nodes kill heavy work:

```bash
pip install pointtree==1.0.1 torch-scatter torch-cluster
```

`torch-scatter` and `torch-cluster` compile against the installed PyTorch, so this can take several minutes.
