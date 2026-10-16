#!/bin/bash
# Task D14's three CPU checks on the full Swiss-Prot test split, from what the probe grid already
# saved -- nothing is retrained or re-embedded.
#
#     cd ~/plm_choice && sbatch scripts/lrz/d14_checks.sbatch        # OUT defaults to ~/d14_checks
#
#   1. pair_redundancy.py      R1.3 / M-34: exact per-protein pair counts on the full split (the
#                              local run had only the 10% sample) and greedy random per-protein caps
#                              10 / 50 / 100, written as (query, target) lists;
#   2. rescore_test_predictions.py  M-40 + M-34: the full100_p10 test predictions re-scored by PIDE
#                              bin (TM-score) and on each capped list (all three targets);
#   3. probe_train_loss.py     R1.10 / M-39: per-epoch train/val loss from the offline wandb runs.
#
# Paths are the grid's own: full100_p10's test.parquet and embeddings are links to
# sprot_pre2024_e1/sets and sprot_pre2024/embeddings_cohort2k; the sha256 of the pairs file is in
# both JSON summaries. Login-node rules forbid this much work there (docs/lrz_setup.md section 3).
#SBATCH -J d14-checks
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
DATA="$DSS/plm_choice_data/data"
PAIRS="$DATA/processed/sprot_pre2024_e1/sets/test.parquet"
OUT="${OUT:-$HOME/d14_checks}"

cd "$HOME/plm_choice"
PY=./.venv/bin/python
echo "host=$(hostname) start=$(date -Is) commit=$(git rev-parse --short HEAD) out=$OUT"
mkdir -p "$OUT"
cmp -s "$PAIRS" "$DATA/processed/full100_p10/sets/test.parquet" \
    || { echo "full100_p10's test.parquet is not sprot_pre2024_e1's"; exit 2; }

$PY scripts/paper/pair_redundancy.py --pairs "$PAIRS" --fai "$DATA/raw/sprot_2024/sprot.fasta.fai" \
    --caps 10 50 100 --tag test_full --out-counts "$OUT/pair_counts" --out-caps "$OUT/pair_caps"
echo "pair_redundancy done $(date -Is)"

$PY scripts/paper/rescore_test_predictions.py --preds "$HOME/plm_choice/models/full100_p10" \
    --pairs "$PAIRS" --emb-dir "$DATA/processed/full100_p10/embeddings" \
    --caps-dir "$OUT/pair_caps" --caps-tag test_full --caps 10 50 100 --out "$OUT/rescore"
echo "rescore done $(date -Is)"

$PY scripts/paper/probe_train_loss.py --models "$HOME/plm_choice/models" \
    --datasets full100_p10 randinit_full100 --out "$OUT/train_loss"
echo "end=$(date -Is)"
