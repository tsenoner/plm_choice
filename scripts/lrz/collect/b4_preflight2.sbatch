#!/bin/bash
# B4 pre-flight v2 — against merged main (7b120fd).
#
# Does double duty:
#   (a) re-verifies the seed/naming/batching contract, since main rewrote 684
#       lines of embedding_generation.py after the v1 smoke test;
#   (b) IS step 3 — the per-arm pooled protein-to-protein cosine degeneracy
#       sweep over ~1000 proteins, for all 13 arms at seed 0.
#
# Side benefit: warms $HF_HOME so the 39-task array does not have 16 tasks
# concurrently downloading the same checkpoints on first start.
#
# Deliberately NOT `set -e` — esm3_open is EXPECTED to fail (uncovered
# GeometricReasoningOriginalImpl parameters make _reinit_weights raise by design).
#SBATCH -J b4-pre2
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/pre2-%j.out
#SBATCH -e /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/pre2-%j.err

set -uo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
SMOKE=$DSS/b4_smoke
export DATA_DIR="$DSS/plm_choice_data/data"
export HF_HOME="$DSS/cache/huggingface"
SPROT="$DATA_DIR/raw/sprot_2024/sprot.fasta"

echo "=========== host/gpu ==========="
hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$HOME/plm_choice" || exit 3
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "FATAL: uv not on PATH"; exit 3; }
echo "repo SHA: $(git rev-parse HEAD)   (expect 7b120fd...)"
uv sync --locked || { echo "FATAL: uv sync failed"; exit 3; }

W="$SMOKE/v2"
rm -rf "$W"; mkdir -p "$W/contract" "$W/arms"

# ~1000 proteins. Every 500th record spreads the sample across the whole file
# rather than taking an alphabetical head, so the length distribution is
# representative — which matters for a degeneracy statistic.
COHORT="$W/arms/cohort.fasta"
awk '/^>/{n++} n%500==1{p=1} /^>/&&n%500!=1{p=0} p' "$SPROT" > "$COHORT"
echo "cohort: $(grep -c '^>' "$COHORT") sequences"
cp "$COHORT" "$W/contract/tiny.fasta"

RUN="uv run python -m data_preparation.embeddings.embedding_generation"

echo ""
echo "########## PART A — contract re-verification on main ##########"
TINY="$W/contract/tiny.fasta"

echo "--- A1: default naming carries the seed ---"
$RUN "$TINY" esm2_8m --random_init --random_seed 0 --embedding_type per_protein \
     --token_budget 4096 > "$W/a1.log" 2>&1; echo "exit=$?"
ls "$W/contract/"

echo "--- A2: re-run same seed must RAISE ---"
$RUN "$TINY" esm2_8m --random_init --random_seed 0 --embedding_type per_protein \
     --token_budget 4096 > "$W/a2.log" 2>&1; echo "exit=$? (non-zero = PASS)"
grep -iE "FileExistsError" "$W/a2.log" | head -2

echo "--- A3: seed 1 differs ---"
$RUN "$TINY" esm2_8m --random_init --random_seed 1 --embedding_type per_protein \
     --token_budget 4096 > "$W/a3.log" 2>&1; echo "exit=$?"

echo "--- A4: token_budget 4096 vs 0 ---"
$RUN "$TINY" esm2_8m --random_init --random_seed 0 --output_hdf5_file "$W/tb0.h5" \
     --embedding_type per_protein --token_budget 0 > "$W/a4a.log" 2>&1; echo "tb0 exit=$?"
$RUN "$TINY" esm2_8m --random_init --random_seed 0 --output_hdf5_file "$W/tb4096.h5" \
     --embedding_type per_protein --token_budget 4096 > "$W/a4b.log" 2>&1; echo "tb4096 exit=$?"

uv run python - "$W" <<'PY'
import h5py, numpy as np, os, sys
W = sys.argv[1]
def load(p):
    with h5py.File(p, "r") as f: return {k: f[k][()] for k in f.keys()}
d = f"{W}/contract"
s0, s1 = f"{d}/random_init_esm2_8m_seed0.h5", f"{d}/random_init_esm2_8m_seed1.h5"
print("A1 seed0 named correctly:", "PASS" if os.path.exists(s0) else "FAIL")
print("A3 seed1 named correctly:", "PASS" if os.path.exists(s1) else "FAIL")
if os.path.exists(s0) and os.path.exists(s1):
    a, b = load(s0), load(s1); ks = sorted(set(a) & set(b))
    print(f"A3 seeds differ ({len(ks)} proteins):",
          "FAIL - IDENTICAL" if all(np.allclose(a[k], b[k]) for k in ks) else "PASS")
t0, t4 = f"{W}/tb0.h5", f"{W}/tb4096.h5"
if os.path.exists(t0) and os.path.exists(t4):
    a, b = load(t0), load(t4); ks = sorted(set(a) & set(b))
    md = max(float(np.abs(a[k].astype("f8") - b[k].astype("f8")).max()) for k in ks)
    print(f"A4 token_budget 4096 vs 0: max|diff|={md:.3e}  ->",
          "PASS" if md < 1e-4 else "FAIL")
PY

echo ""
echo "########## PART B — per-arm degeneracy sweep (step 3) ##########"
MODELS=(esm1b esm2_8m esm2_35m esm2_150m esm2_650m esm2_3b esm3_open
        esmc_300m esmc_600m ankh_base ankh_large prot_t5 prost_t5)
for M in "${MODELS[@]}"; do
  echo ""
  echo "--- arm: $M ---"
  /usr/bin/time -f "  wall=%E maxRSS=%MkB" \
    $RUN "$COHORT" "$M" --random_init --random_seed 0 \
      --output_hdf5_file "$W/arms/random_init_${M}_seed0.h5" \
      --embedding_type per_protein --token_budget "${TOKEN_BUDGET:-4096}" \
      > "$W/arm_${M}.log" 2>&1
  RC=$?
  echo "  exit=$RC"
  [ $RC -ne 0 ] && tail -4 "$W/arm_${M}.log" | sed 's/^/    /'
done

echo ""
echo "########## PART C — degeneracy verdicts ##########"
ls -la "$W/arms/"*.h5 2>/dev/null | awk '{print "  ",$5,$9}'
for H in "$W/arms"/*.h5; do
  [ -e "$H" ] || continue
  uv run python "$HOME/preflight_degeneracy_check.py" "$H" 2>&1 | \
      grep -E "^=== |RAW cosine|CENTRED cosine|participation|VERDICT" | sed 's/^/  /'
done

echo ""
echo "=========== done ==========="; date
