#!/bin/bash
# UMAP of a superkingdom-stratified 50k-protein subsample (R1.3), CPU only.
# See scripts/paper/umap_subsample.py.
#
#     cd ~/plm_choice && mkdir -p logs
#     CODE=<checkout> sbatch <checkout>/scripts/lrz/umap_subsample.sbatch
#
# umap-learn is not a project dependency, so the job builds a throwaway venv on node-local disk
# with pinned versions (the ones the script was tested with) instead of touching the project's
# lock. The uv cache goes there too: HOME is quota-bound and DSS forbids many small files.
#SBATCH -J umap-subsample
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
CODE="${CODE:-$HOME/plm_choice}"
OUT="${OUT:-$DSS/d14_open/umap}"
ARMS="${ARMS:-prott5 esmc_600m random_1024}"
SCRATCH="${TMPDIR:-/tmp}/umap_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

export UV_CACHE_DIR="$SCRATCH/uvcache" NUMBA_CACHE_DIR="$SCRATCH/numba"
UV="$HOME/.local/bin/uv"
"$UV" venv -q --python 3.12 "$SCRATCH/venv"
"$UV" pip install -q --python "$SCRATCH/venv/bin/python" \
    umap-learn==0.5.12 numba==0.68.0 scikit-learn==1.6.0 polars==1.31.0 h5py==3.12.1 \
    matplotlib==3.10.0 scipy==1.14.1

echo "code=$CODE ($(git -C "$CODE" rev-parse --short HEAD)) arms=$ARMS host=$(hostname)"
cd "$CODE"
# shellcheck disable=SC2086
"$SCRATCH/venv/bin/python" scripts/paper/umap_subsample.py \
    --sprot-json "$DSS/plm_choice_data/data/raw/sprot_2024/sprot.json" \
    --emb-dir "$DSS/plm_choice_data/data/processed/full100_p10/embeddings" \
    --arms $ARMS --out "$OUT"
