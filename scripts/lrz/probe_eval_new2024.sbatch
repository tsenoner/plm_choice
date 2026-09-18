#!/bin/bash
# Score the retrained probes on the New2024 set (R1.1: proteins added to Swiss-Prot AFTER the
# models' training cut-offs), without retraining anything.
#
#     cd ~/plm_choice && mkdir -p logs && sbatch scripts/lrz/probe_eval_new2024.sbatch
#
# ---------------------------------------------------------------------------
# WHY A COPY OF THE CELL RATHER THAN RE-POINTING IT. evaluate.py resolves the embedding file
# and the data dir from the trained cell's wandb config; editing that in place would leave the
# cell describing New2024 instead of the cohort it was trained and tested on. Instead each cell
# is mirrored into models/new2024_e1/... with only the best checkpoint plus a hparams.yaml
# naming the New2024 paths -- there is no wandb dir there, so evaluate.py's documented fallback
# to hparams.yaml is what gets used, and the original cell is never touched.
#
# The test file is named new2024.parquet on purpose: evaluate.py derives the test-set name from
# the file stem and writes test_<stem>_<ckpt>_metrics.txt, so calling it test.parquet would
# collide with the Swiss-Prot test metrics of the same cell.
#
# 14 arms, not 15: New2024 has no random_1024 (the repo's random-embedding generator is
# unseeded, so a New2024 noise arm would not be reproducible). clean/esm1b cover the 1,173
# proteins <=1022 aa; the other arms cover 1,225, and the pair table is already restricted to
# the <=1022 set so every arm is scored on the same pairs.
#
# N IS SMALL: 5,466 pairs over 673 proteins (2,843 with fident/hfsp, 5,052 with alntmscore).
# Size any claim to that, and expect wide CIs next to the Swiss-Prot test numbers.
#
# CPU only: inference over a few thousand pairs needs no GPU, and the GPU budget is the probe
# grid's. One job walks every cell; each cell takes seconds.
# ---------------------------------------------------------------------------
#SBATCH -J probe-new2024
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
NEW="$DSS/plm_choice_data/data/processed/2024_new_e1"
SRC_DATASET=sprot_pre2024_e1_sub10
DST_DATASET=new2024_e1
TEST="$NEW/sets/new2024.parquet"

cd "$HOME/plm_choice"
PY=./.venv/bin/python
MODELS="$HOME/plm_choice/models"

[ -f "$TEST" ] || { echo "missing $TEST"; exit 2; }
mapfile -t ARMS < <(find -L "$NEW/embeddings" -maxdepth 1 -name '*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
echo "arms: ${ARMS[*]}"
[ "${#ARMS[@]}" -eq 14 ] || { echo "expected 14 New2024 arms, found ${#ARMS[@]}"; exit 2; }

ok=0; skipped=0; failed=0
for model in fnn euclidean; do
  for param in fident alntmscore hfsp; do
    for arm in "${ARMS[@]}"; do
      src="$MODELS/$SRC_DATASET/$model/$param/$arm"
      dst="$MODELS/$DST_DATASET/$model/$param/$arm"
      if [ ! -f "$src/training_complete.txt" ]; then
        echo "skip (not trained yet): $model/$param/$arm"; skipped=$((skipped + 1)); continue
      fi
      mkdir -p "$dst/checkpoints"
      if [ "$model" != euclidean ]; then
        ckpt=$(ls "$src"/checkpoints/best-*.ckpt 2>/dev/null | head -1) || true
        if [ -z "${ckpt:-}" ]; then
          echo "skip (no checkpoint): $model/$param/$arm"; skipped=$((skipped + 1)); continue
        fi
        cp -f "$ckpt" "$dst/checkpoints/"
      fi
      cat > "$dst/hparams.yaml" <<EOF
model_type: $model
param_name: $param
embedding_file: $NEW/embeddings/$arm.h5
data_dir: $NEW/sets
batch_size: 1024
seed: 42
dataset: $DST_DATASET
trained_from: $src
EOF
      # evaluate.py catches its own exceptions and exits 0, so assert on the artifact.
      $PY src/evaluation/evaluate.py --run_dir "$dst" --test_file "$TEST" \
          --n_bootstrap 1000 --force_recompute || true
      if ls "$dst"/evaluation_results/test_new2024_*_metrics.txt >/dev/null 2>&1; then
        ok=$((ok + 1))
      else
        echo "FAILED: $model/$param/$arm"; failed=$((failed + 1))
      fi
    done
  done
done

echo "new2024 evaluation: ok=$ok skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ]
