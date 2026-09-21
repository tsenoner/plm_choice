#!/bin/bash
# Recompute the per-split pair counts that Supplementary Methods states and nothing
# regenerated (audit, 2026-09-21). Cheap -- a column-null scan over ~1 GB of parquet --
# but it reads from GPFS, where a login-node scan of these files measured minutes rather
# than seconds, so it runs as a job.
#
#     sbatch scripts/lrz/pair_table_counts.sbatch
#
#SBATCH -J pair-counts
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
cd "$HOME/plm_choice"

./.venv/bin/python scripts/paper/pair_table_counts.py \
    --sets-dir "$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets" \
    --out "$DSS/pair_table_counts/sprot_pre2024_e1.json"
