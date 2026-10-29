#!/bin/bash
# Closed-form ridge on |a-b| and (a-b)^2 (R1.8), one array task per embedding arm, all three
# targets, on the main grid's pairs (all training pairs). See scripts/paper/ridge_diff_probe.py.
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --array=0-14 <checkout>/scripts/lrz/ridge_diff_probe.sbatch
#
# CODE is the checkout whose scripts/paper/ridge_diff_probe.py runs (default ~/plm_choice); the
# venv is always ~/plm_choice/.venv. Each task asserts that its replayed Euclidean Spearman
# equals the paper's full100_p10 Euclidean cell, so the pairs are proven to be the paper's.
#SBATCH -J ridge-diff
#SBATCH -p lrz-hgx-a100-80x4,lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-v100x2,lrz-dgx-1-v100x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
EMB="$DSS/plm_choice_data/data/processed/full100_p10/embeddings"
SETS="$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets"
OUT="${OUT:-$DSS/d14_open/ridge}"
CODE="${CODE:-$HOME/plm_choice}"
PY="$HOME/plm_choice/.venv/bin/python"

mapfile -t ARMS < <(find -L "$EMB" -maxdepth 1 -name '*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
ARM=${ARMS[${SLURM_ARRAY_TASK_ID:?set --array}]}

# Stage off GPFS: reading one HDF5 dataset per protein is metadata-bound there.
H5="$EMB/$ARM.h5"
for CANDIDATE in "/dev/shm/ridge_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}" "${TMPDIR:-/tmp}/ridge_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; do
    if mkdir -p "$CANDIDATE" 2>/dev/null && cp -L "$H5" "$CANDIDATE/$ARM.h5" 2>/dev/null; then
        H5="$CANDIDATE/$ARM.h5"; STAGE="$CANDIDATE"; trap 'rm -rf "$STAGE"' EXIT
        echo "staged $ARM.h5 to $CANDIDATE"; break
    fi
    rm -rf "$CANDIDATE" 2>/dev/null || true
done

echo "arm=$ARM code=$CODE ($(git -C "$CODE" rev-parse --short HEAD)) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cd "$CODE"
$PY scripts/paper/ridge_diff_probe.py --embedding "$H5" --sets-dir "$SETS" --out "$OUT" \
    --check-metrics "$HOME/plm_choice/models/full100_p10/euclidean"
