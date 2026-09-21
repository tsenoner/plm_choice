#!/bin/bash
#SBATCH --job-name=slice_union
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/dss/dsshome1/02/ge45ted2/c1_strat/slices_union/slice_union_%j.log
#SBATCH --error=/dss/dsshome1/02/ge45ted2/c1_strat/slices_union/slice_union_%j.log

set -euo pipefail
echo "host=$(hostname) start=$(date -Is)"
srun --cpu-bind=none /dss/dsshome1/02/ge45ted2/plm_choice/.venv/bin/python -u \
  /dss/dsshome1/02/ge45ted2/c1_strat/slices_union/slice_union.py
echo "end=$(date -Is)"
