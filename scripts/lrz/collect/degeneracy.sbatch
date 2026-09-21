#!/bin/bash
# B4 degeneracy gate — the pre-flight check the plan required before spending
# GPU-days, run after the fact because the array completed unattended.
#
# CPU only: the embeddings already exist. The pretrained cohort2k arms are run in
# the same job as POSITIVE CONTROLS, so each untrained verdict is read against its
# own trained twin rather than against an absolute threshold — transformers are
# anisotropic and absolute cosine thresholds false-alarm (a healthy synthetic
# control scores 0.97; the real prostt5 arm scores 0.647).
#SBATCH -J b4-degen
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/degen-%j.out
#SBATCH -e /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/degen-%j.err

set -uo pipefail
D=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
OUT=$D/b4_smoke
mkdir -p "$OUT"
cd "$HOME/plm_choice" || exit 3
export PATH="$HOME/.local/bin:$PATH"
hostname; date; echo "repo SHA: $(git rev-parse --short HEAD)"

echo ""
echo "############## UNTRAINED (B4 random-init) ##############"
uv run python ~/degeneracy.py \
    "$D/plm_choice_data/data/processed/sprot_pre2024_subset/embeddings_random_init/*.h5" \
    --json "$OUT/degeneracy_random_init.json"
RC_UNTRAINED=$?

echo ""
echo "############## PRETRAINED cohort2k (positive controls) ##############"
uv run python ~/degeneracy.py \
    "$D/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k/*.h5" \
    --json "$OUT/degeneracy_pretrained.json"

echo ""
echo "untrained gate exit=$RC_UNTRAINED  (0 = no arm failed)"
date
