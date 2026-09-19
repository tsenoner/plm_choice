#!/bin/bash
# Score already-trained probe cells on an EXTERNAL test set, without retraining anything.
#
#   SETS=$DSS/plm_choice_data/data/processed/exp_tm/sets TEST=exptm_matched \
#   EMB=$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k \
#   DST=exptm_matched TARGETS=alntmscore sbatch scripts/lrz/probe_eval_external.sbatch
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS — the fairness question. The probes are trained on Foldseek's
# `alntmscore`, computed over AlphaFold models, and that is also what they are tested on.
# Scoring them against a TM-score from experimental structures is a different test, and it
# is only fair if the two numbers are normalised the same way: Foldseek normalises
# alntmscore by the ALIGNMENT SPAN, while a whole-chain TM-score normalises by chain
# length. On the 10,740 experimental pairs that difference alone is +0.064 TM -- larger
# than most of the effects in this paper. So this runs twice:
#   exptm_matched     target = US-align TM normalised by the SHORTER chain (comparable)
#   exptm_wholechain  target = US-align TM normalised by the LONGER chain (not comparable)
# and the gap between them is the size of the normalisation artifact, reported, not hidden.
#
# The cell is mirrored (best checkpoint + a hparams.yaml naming the real embeddings and the
# external test file) rather than re-pointed, so the trained cell keeps describing the
# cohort it was trained on. evaluate.py takes its documented hparams.yaml fallback because
# the mirror has no wandb directory. It also catches its own exceptions and exits 0, so
# success is asserted on the metrics file.
# ---------------------------------------------------------------------------
#SBATCH -J probe-eval-ext
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

SRC_DATASET="${SRC_DATASET:-sprot_pre2024_e1_sub10}"
SETS="${SETS:?set SETS to the directory holding the external test parquet}"
TEST="${TEST:?set TEST to the test parquet basename, without .parquet}"
EMB="${EMB:?set EMB to the embeddings directory the cells were trained on}"
DST="${DST:?set DST to the destination dataset name under models/}"
TARGETS="${TARGETS:-fident alntmscore hfsp}"

cd "$HOME/plm_choice"
PY=./.venv/bin/python
MODELS="$HOME/plm_choice/models"
TESTFILE="$SETS/$TEST.parquet"
[ -f "$TESTFILE" ] || { echo "missing $TESTFILE"; exit 2; }

ok=0; skipped=0; failed=0
for model in fnn euclidean; do
  for param in $TARGETS; do
    for src in "$MODELS/$SRC_DATASET/$model/$param"/*; do
      [ -d "$src" ] || continue
      arm=$(basename "$src")
      [ -f "$src/training_complete.txt" ] || { echo "skip (untrained): $model/$param/$arm"; skipped=$((skipped+1)); continue; }
      dst="$MODELS/$DST/$model/$param/$arm"
      mkdir -p "$dst/checkpoints"
      if [ "$model" != euclidean ]; then
        ckpt=$(ls "$src"/checkpoints/best-*.ckpt 2>/dev/null | head -1) || true
        [ -n "${ckpt:-}" ] || { echo "skip (no checkpoint): $model/$param/$arm"; skipped=$((skipped+1)); continue; }
        cp -f "$ckpt" "$dst/checkpoints/"
      fi
      cat > "$dst/hparams.yaml" <<EOF
model_type: $model
param_name: $param
embedding_file: $EMB/$arm.h5
data_dir: $SETS
batch_size: 1024
seed: 42
dataset: $DST
trained_from: $src
EOF
      $PY src/evaluation/evaluate.py --run_dir "$dst" --test_file "$TESTFILE" \
          --n_bootstrap 1000 --force_recompute || true
      if ls "$dst"/evaluation_results/test_${TEST}_*_metrics.txt >/dev/null 2>&1; then
        ok=$((ok+1))
      else
        echo "FAILED: $model/$param/$arm"; failed=$((failed+1))
      fi
    done
  done
done

echo "$DST: ok=$ok skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ]
