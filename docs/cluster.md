# Running on PACE-ICE

Connect the GT VPN first, then `ssh <gt-username>@login-ice.pace.gatech.edu`. Home is small (30 GB), so everything goes in scratch, which `~/scratch` links to. Scratch can be purged, so copy results you want to keep somewhere else.

Login nodes kill heavy work, so installs and runs go through Slurm jobs. Run every command below from the repo root unless it says otherwise.

## One-time setup

### 1. Clone

To send changes back, fork the repo on GitHub first and clone your fork instead; pushing to it from ICE needs a GitHub personal access token as the password.

```bash
cd ~/scratch
git clone https://github.com/nisulay/pointsam-training.git
cd pointsam-training
```

### 2. Install the environment

```bash
mkdir -p slurm_outs && sbatch scripts/slurm/install_env.sbatch
```

It downloads Point-SAM and its weights into `thirdparty/`, builds the conda environment in `env/`, compiles torkit3d and runs the tests. The log is `slurm_outs/install_<jobid>.out`, ending in `INSTALL_DONE`. Rerunning it skips finished steps.

### 3. Get the data

Everything is in one Hugging Face dataset, [haag-lidar/wytham-isprs-tiles](https://huggingface.co/datasets/haag-lidar/wytham-isprs-tiles): the authors' original scan in `raw/`, and the tiles and crops we cut from it. It is public, so no login is needed. The data is from Cherlet et al. (2026), CC BY 4.0, https://doi.org/10.5281/zenodo.16875688.

`notebooks/00_dataset.ipynb` does this step with pictures and checks, and is the place to look to understand the data. On ICE the two commands below do the same thing from a terminal.

To train and score, the tiles and crops are enough (0.9 GB). From the repo root:

```bash
env/bin/hf download haag-lidar/wytham-isprs-tiles --repo-type dataset --local-dir data/wytham_isprs --include "tiles/*" --include "crops/*" --include summary.json
```

To rebuild them from the original scan, get `raw/` as well (3.3 GB):

```bash
env/bin/hf download haag-lidar/wytham-isprs-tiles --repo-type dataset --local-dir data/wytham_isprs --include "raw/*"
```

Then build tiles and crops on a compute node (only needed for a rebuild):

```bash
cd ~/scratch/pointsam-training
salloc -p ice-cpu -c 4 --mem=32G -t 01:00:00
module load anaconda3/2023.03 && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ./env
python scripts/prepare_data.py
exit
```

It prints one line per strip and writes `data/wytham_isprs/`. Expect 182 train, 56 val and 56 test crops.

### 4. Add TreeX (needed by `click_methods: treex`)

```bash
sbatch scripts/slurm/install_treex.sbatch
```

The log is `slurm_outs/install/install_treex_<jobid>.out`, ending in `TREEX_INSTALL_DONE`. Use this job rather than a plain `pip install pointtree`: that changes pandas, scipy and psutil and breaks the environment. The job holds everything else at its installed version.

## Run a notebook

```bash
bash scripts/slurm/submit.sh notebooks/02_train.ipynb configs/experiment/smoke.yaml
```

This gives the job a readable name and writes its log to a folder for what the run is testing, such as `slurm_outs/smoke_tests/smoke_<jobid>.out`. The finished notebook, with its outputs, is saved to `results/notebooks/`. Anything after the config goes to `sbatch`, for example `--dependency=afterok:<jobid>` to start after another job succeeds.

- **Check and cancel:** `squeue -u $USER`, `scancel <jobid>`. Watch a run live with `tail -F <log path>`.
- **Order:** `smoke.yaml` first (4 steps), then `isprs_lora.yaml` (about 20 minutes on an H100), then `03_benchmark.ipynb` with `configs/experiment/diagnose_val.yaml`, which scores the val tiles and saves every mask.
- **Look at one tile:** copy `results/diagnose_val/*_masks.npz` and `*_pred.npy` to the same folder on a Mac and open `04_inference_steps.ipynb` there, or run it live in OnDemand Jupyter.
- **Test strip:** score it once, after choosing the checkpoint (`benchmark.yaml`). Tune on the val strip only.
- **New experiment:** copy a config in `configs/experiment/`, change it, and add a line for it in `submit.sh` so its log goes to the right folder. Unknown configs go to `slurm_outs/other/`.
- **GPU type:** `run_notebook.sbatch` asks for an H100. To see what's free, run `sinfo -p ice-gpu -o "%G %t %D"`; to use another NVIDIA GPU, change `--gres` there. AMD GPUs won't work.
- **Time limit:** `ice-gpu` jobs can run for up to 16 hours; `run_notebook.sbatch` asks for 3.

## Interactive notebooks

To run cells by hand on a GPU, open Jupyter on a GPU node through ICE's OnDemand portal, not on the login node. Pick the config in the first cell:

```python
import os
os.environ["TREESEG_CONFIG"] = "configs/experiment/isprs_lora.yaml"
```

## Copying from a Mac instead of cloning

If you already have the code and prepared data on a Mac, rsync them over (VPN on, from the folder that holds `pointsam-training`):

```bash
rsync -avh --progress --exclude .git --exclude env --exclude results --exclude thirdparty --exclude slurm_outs --exclude __pycache__ --exclude .pytest_cache --exclude inspect pointsam-training <gt-username>@login-ice.pace.gatech.edu:~/scratch/
```

**Never add `--delete` or `--delete-excluded`.** They remove anything on ICE that this copy leaves out, including `env/`, `thirdparty/` and `results/`.
