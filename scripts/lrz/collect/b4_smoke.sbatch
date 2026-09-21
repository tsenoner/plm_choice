#!/bin/bash
# B4 pre-flight smoke test — verifies the seed/naming/batching contract on a GPU
# node before committing ~200 GPU-h to the 39-task array.
#
# Deliberately NOT `set -e`: several steps are EXPECTED to fail, and the point of
# the test is to observe that they fail in the right way.
#SBATCH -J b4-smoke
#SBATCH -p lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH -o /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/smoke-%j.out
#SBATCH -e /dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/smoke-%j.err

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
echo "repo SHA: $(git rev-parse HEAD)"

echo "=========== uv sync ==========="
uv sync --locked || { echo "FATAL: uv sync failed"; exit 3; }

# --- tiny cohort: first 20 records of sprot.fasta -------------------------
rm -rf "$SMOKE/work"
mkdir -p "$SMOKE/work/default" "$SMOKE/work/tb"
TINY="$SMOKE/work/default/tiny.fasta"
awk '/^>/{n++} n>20{exit} {print}' "$SPROT" > "$TINY"
echo "tiny cohort: $(grep -c '^>' "$TINY") sequences"

RUN="uv run python -m data_preparation.embeddings.embedding_generation"

echo ""
echo "=========== T1: default naming carries the seed ==========="
$RUN "$TINY" esm2_8m --random_init --random_seed 0 \
     --embedding_type per_protein --token_budget 4096 > "$SMOKE/work/t1.log" 2>&1
echo "exit=$?"
ls -la "$SMOKE/work/default/"

echo ""
echo "=========== T2: re-running the SAME seed must RAISE ==========="
$RUN "$TINY" esm2_8m --random_init --random_seed 0 \
     --embedding_type per_protein --token_budget 4096 > "$SMOKE/work/t2.log" 2>&1
echo "exit=$? (NON-ZERO IS THE PASS CONDITION)"
grep -iE "FileExistsError|already open|unable to (create|lock)" "$SMOKE/work/t2.log" | head -3

echo ""
echo "=========== T3: seed 1 produces its own file ==========="
$RUN "$TINY" esm2_8m --random_init --random_seed 1 \
     --embedding_type per_protein --token_budget 4096 > "$SMOKE/work/t3.log" 2>&1
echo "exit=$?"
ls -la "$SMOKE/work/default/"

echo ""
echo "=========== T4: token_budget 4096 vs 0 (must be identical) ==========="
$RUN "$TINY" esm2_8m --random_init --random_seed 0 \
     --output_hdf5_file "$SMOKE/work/tb/tb0.h5" \
     --embedding_type per_protein --token_budget 0 > "$SMOKE/work/t4a.log" 2>&1
echo "tb0 exit=$?"
$RUN "$TINY" esm2_8m --random_init --random_seed 0 \
     --output_hdf5_file "$SMOKE/work/tb/tb4096.h5" \
     --embedding_type per_protein --token_budget 4096 > "$SMOKE/work/t4b.log" 2>&1
echo "tb4096 exit=$?"

echo ""
echo "=========== T5: numeric verdicts ==========="
uv run python - <<'PY'
import h5py, numpy as np, os, sys
W = "/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/b4_smoke/work"
def load(p):
    with h5py.File(p, "r") as f:
        return {k: f[k][()] for k in f.keys()}

d = f"{W}/default"
print("files produced by default naming:", sorted(os.listdir(d)))

s0p, s1p = f"{d}/random_init_esm2_8m_seed0.h5", f"{d}/random_init_esm2_8m_seed1.h5"
print("T1 seed0 named correctly:", "PASS" if os.path.exists(s0p) else "FAIL")
print("T3 seed1 named correctly:", "PASS" if os.path.exists(s1p) else "FAIL")

if os.path.exists(s0p) and os.path.exists(s1p):
    a, b = load(s0p), load(s1p)
    keys = sorted(set(a) & set(b))
    same = all(np.allclose(a[k], b[k]) for k in keys)
    print(f"T3 seeds differ ({len(keys)} proteins):", "FAIL - SEEDS IDENTICAL" if same else "PASS")
    k0 = keys[0]
    print(f"   cos(seed0,seed1) on {k0}: "
          f"{float(a[k0]@b[k0]/(np.linalg.norm(a[k0])*np.linalg.norm(b[k0]))):.4f}")

t0, t4 = f"{W}/tb/tb0.h5", f"{W}/tb/tb4096.h5"
if os.path.exists(t0) and os.path.exists(t4):
    a, b = load(t0), load(t4)
    keys = sorted(set(a) & set(b))
    md = max(float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max()) for k in keys)
    exact = all(np.array_equal(a[k], b[k]) for k in keys)
    print(f"T4 token_budget 4096 vs 0: max|diff|={md:.3e}  bitwise_identical={exact}")
    print("T4 verdict:", "PASS" if md < 1e-5 else "FAIL")
else:
    print("T4 verdict: FAIL - one or both files missing")
PY

echo ""
echo "=========== T6: does a CPU torch.Generator work on CUDA tensors? ==========="
uv run python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
t = torch.empty(4, 4, device="cuda")
g = torch.Generator(); g.manual_seed(0)
try:
    torch.nn.init.normal_(t, mean=0.0, std=0.02, generator=g)
    print("T6: CPU generator on CUDA tensor SUCCEEDED -> blocker #1 REFUTED")
except RuntimeError as e:
    print("T6: CPU generator on CUDA tensor RAISED -> blocker #1 CONFIRMED")
    print("    ", e)
PY

echo ""
echo "=========== T7: the real thing — esmc_300m --random_init ==========="
timeout 900 $RUN "$TINY" esmc_300m --random_init --random_seed 0 \
     --output_hdf5_file "$SMOKE/work/tb/esmc.h5" \
     --embedding_type per_protein --token_budget 4096 > "$SMOKE/work/t7.log" 2>&1
echo "esmc_300m exit=$? (non-zero => 9 of 39 array tasks would die here)"
tail -25 "$SMOKE/work/t7.log"

echo ""
echo "=========== done ==========="
date
