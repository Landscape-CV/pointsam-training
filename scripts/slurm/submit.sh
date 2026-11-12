#!/bin/bash
# Submit a notebook run under a readable name, with its log in a folder for what the run is testing:
#   slurm_outs/<folder>/<name>_<jobid>.out
#
# From the repo root:
#   bash scripts/slurm/submit.sh notebooks/02_train.ipynb configs/experiment/smoke.yaml
# Anything after the config goes to sbatch, for example --dependency=afterok:<jobid>
# See names and states:  squeue -u $USER -o "%.10i %.24j %.8T %.10M %R"
# To add a kind of run: add a line to the case below.

set -e
NOTEBOOK="$1"
CONFIG="$2"
shift 2

case "$(basename "$NOTEBOOK" .ipynb)/$(basename "$CONFIG" .yaml)" in
    */smoke)                  FOLDER=smoke_tests;        NAME=smoke ;;
    */isprs_lora)             FOLDER=augmentation_test;  NAME=with_augmentation ;;
    */isprs_lora_noaug)       FOLDER=augmentation_test;  NAME=no_augmentation ;;
    */isprs_lora_lr5e5)       FOLDER=learning_rate_test; NAME=lr_5e-5 ;;
    */isprs_lora_lr2e4)       FOLDER=learning_rate_test; NAME=lr_2e-4 ;;
    */isprs_lora_batch4)      FOLDER=batch_size_test;    NAME=batch_4 ;;
    03_benchmark/benchmark)   FOLDER=test_scoring;       NAME=f1_benchmark ;;
    03_benchmark/benchmark_clicks) FOLDER=test_scoring;  NAME=click_comparison ;;
    */diagnose_val)           FOLDER=diagnosis;          NAME=diagnose_val ;;
    */negative_clicks)        FOLDER=diagnosis;          NAME=negative_clicks ;;
    */isprs_lora_trunk)       FOLDER=target_test;        NAME=trunk_targets ;;
    */isprs_lora_whole)       FOLDER=target_test;        NAME=whole_targets ;;
    */whole_targets_val)      FOLDER=target_test;        NAME=whole_targets_val ;;
    */trunk_targets_val)      FOLDER=target_test;        NAME=trunk_targets_val ;;
    */mask_experiments)       FOLDER=diagnosis;          NAME=mask_experiments ;;
    *)                        FOLDER=other;              NAME="$(basename "$CONFIG" .yaml)" ;;
esac

# Slurm doesn't create the log folder, and a job with no folder fails without a log
mkdir -p "slurm_outs/$FOLDER" results/notebooks

# atl1-1-03-013-8-0 lost its GPU mid-job on 30 Oct 2026
sbatch -J "$NAME" -o "slurm_outs/$FOLDER/${NAME}_%j.out" -c 4 --exclude=atl1-1-03-013-8-0 "$@" \
    scripts/slurm/run_notebook.sbatch "$NOTEBOOK" "$CONFIG"
echo "name: $NAME  log: slurm_outs/$FOLDER/${NAME}_<jobid>.out"
