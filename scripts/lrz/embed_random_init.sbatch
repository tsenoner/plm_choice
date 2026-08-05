#!/bin/bash
# Random-init frozen baseline embeddings (Track B4) on LRZ AI Systems.
#
# Submit as an array over the 13 model x 3 seed grid = 39 tasks (0-38):
#
#     cd ~/plm_choice
#     mkdir -p logs            # REQUIRED: see the -o/-e note below
#     uv sync --locked         # once, not from 39 concurrent tasks
#     sbatch --array=0-38 scripts/lrz/embed_random_init.sbatch
#
# Validate without queueing anything first:
#
#     sbatch --test-only --array=0-38 scripts/lrz/embed_random_init.sbatch
#
# `--test-only` does NOT check the -o/-e paths. logs/ is gitignored, so a fresh
# clone does not have it, and Slurm will not create it: every task would be
# allocated after the multi-day queue wait and then die at step launch with
# "Unable to open file logs/..." — written nowhere readable, because both stdout
# and stderr point into the missing directory. Hence the mkdir above.
#
# The script derives its own bounds and refuses an out-of-range index rather
# than silently mis-pairing a model with the wrong seed.
#
# ---------------------------------------------------------------------------
# PARTITION / QOS — do not "restore" mcml here.
# The resubmission plan originally specified `-p mcml-hgx-h100-94x4 --qos=mcml`.
# Verified on the cluster 2026-07-29: this account's only Slurm association is
#     Cluster=lrz  Account=default  QOS=cpu,gpu,mig
# and all four mcml-* partitions are AllowQos=mcml. Those jobs are rejected at
# submission — verified: `allocation failure: Invalid qos specification`.
#
# Usable GPU partitions under QOS `gpu`, with start estimates measured by
# `sbatch --test-only` for one 8h job on 2026-08-04:
#
#     lrz-v100x2           2026-08-07  (queue  31)  16GB Volta — NO bf16
#     lrz-hgx-a100-80x4    2026-08-10  (queue  43)  80GB, bf16      <- default
#     lrz-hgx-h100-94x4    2026-08-15  (queue 309)  fastest, most contended
#     lrz-dgx-a100-80x8    2026-08-24  (queue 108)  avoid
#
# A100-80x4 is the balance: bf16 support with roughly a third of the H100 wait.
# Override per submission with `sbatch -p <partition> ...`.
#
# Shorter --time backfills much sooner. If an arm demonstrably finishes in two
# hours, asking for two hours will start days earlier than asking for eight.
# ---------------------------------------------------------------------------
#SBATCH -J plm-randinit
#SBATCH -p lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1                 # MANDATORY: without it the job sits PD forever (QOSMinGRES)
#SBATCH --time=08:00:00              # MANDATORY: default is 1h, QOS caps at 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL         # never poll squeue in a loop — that risks a user-ID ban
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

# Bulk data lives outside the git worktree on purpose: `git clean -xdf` removes
# untracked paths and -x targets gitignored ones precisely, which is exactly
# what data/ and models/ are.
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
export DATA_DIR="${DATA_DIR:-$DSS/plm_choice_data/data}"

# Under sprot_pre2024_SUBSET, because that is the cohort the probe trains on —
# every published cell used it (11.3M pairs); the full splits are 10x larger, and
# giving the untrained arm 10x the probe budget biases the comparison toward it,
# understating the very quantity being claimed.
#
# NOTE these land in embeddings_random_init/, not embeddings/. run_experiments.py
# globs "<data_dir>/embeddings/*.h5" and needs a sibling sets/, so the files must
# be moved or symlinked into embeddings/ before the probe grid runs. Kept separate
# here so a half-finished array cannot be picked up by an unrelated probe run.
#
# scripts/run_ivan_pipeline.sh's RANDOM_INIT_OUT_DIR resolves to this same path via
# PROBE_COHORT_DIR. The two entry points to the random-init arm must be changed together.
OUT_DIR="${OUT_DIR:-$DATA_DIR/processed/sprot_pre2024_subset/embeddings_random_init}"

# Keep the HuggingFace cache off the 100 GB backed-up HOME.
export HF_HOME="${HF_HOME:-$DSS/cache/huggingface}"

# The 13 MODEL_CONFIGS keys. Per src/visualization/plm_constants.py:39-41 the
# paper's remaining arms are derivatives of these: clean -> ESM-1 (covered by the
# new esm1b entry) and prottucker -> ProtT5 (covered by prot_t5). random_1024 is
# an i.i.d. noise floor, not a pLM, so it needs no untrained twin. Every arm in
# the comparison therefore now has an architectural twin.
#
# esm1b caps at 1022 residues (absolute position embeddings, unlike the rotary
# ESM-2s); the script clamps --max_seq_len to that automatically. That arm
# therefore covers strictly fewer proteins than the other twelve — join on the
# intersection when building any cross-arm table.
#
# esm3_open / esmc_300m / esmc_600m expose their own encode()/logits() API rather
# than a tokenizer + forward pass, so --token_budget does NOT apply to them; they
# run one sequence per pass (the run now warns when this happens). That is 9 of
# the 39 tasks, and 8h is not enough for them — submit those separately with a
# longer --time, e.g.:
#
#     sbatch --time=24:00:00 --array=6,7,8,19,20,21,32,33,34 scripts/lrz/embed_random_init.sbatch
#     sbatch --array=0-5,9-18,22-31,35-38 scripts/lrz/embed_random_init.sbatch
# Hand-typed for readability, then CHECKED against the registry below rather than
# trusted. A stale hand-maintained arm list is exactly the defect the new esm1b
# entry fixes (and src/data_preparation/embeddings/batch_embedding_generation.sh
# still carries a list containing "prot_t5_xl", which is not a valid key at all).
# Preserving the mechanism one level up in bash would just move the drift.
MODELS=(
    esm1b
    esm2_8m esm2_35m esm2_150m esm2_650m esm2_3b
    esm3_open esmc_300m esmc_600m
    ankh_base ankh_large
    prot_t5 prost_t5
)
SEEDS=(0 1 2)   # D-6: report mean±sd over exactly these

N_MODELS=${#MODELS[@]}
N_SEEDS=${#SEEDS[@]}
N_TASKS=$((N_MODELS * N_SEEDS))
TASK="${SLURM_ARRAY_TASK_ID:-0}"

# An oversized --array would otherwise index SEEDS out of bounds. Under `set -u`
# that aborts, but only after the job has been queued and scheduled; failing
# here with the real numbers is far easier to read in a log.
if [ "$TASK" -ge "$N_TASKS" ]; then
    echo "ERROR: array index $TASK is out of range: $N_MODELS models x $N_SEEDS seeds = $N_TASKS tasks (0-$((N_TASKS - 1)))" >&2
    exit 2
fi

MODEL="${MODELS[$((TASK % N_MODELS))]}"
SEED="${SEEDS[$((TASK / N_MODELS))]}"

# A superset of the ~481k proteins the subset splits reference. Embedding extra
# proteins is scientifically harmless (~4% more work, and the probe only reads the
# ids its splits name) — what matters is which SPLITS the probe trains on, which
# is OUT_DIR's business above, not this file's.
#
# VERIFY THIS PATH BEFORE SUBMITTING: it appears nowhere else in the repo, and the
# check that would catch a typo (embedding_generation.py exits 1 on a missing
# FASTA) runs *after* the GPU allocation — so all 39 tasks would each burn an A100
# slot to print "FASTA file not found".
FASTA="${FASTA:-$DATA_DIR/raw/sprot_2024/sprot.fasta}"

# Explicit per-seed output path. The default would also carry the seed now, but
# being explicit here means a rename upstream cannot silently collapse the grid.
OUT="$OUT_DIR/random_init_${MODEL}_seed${SEED}.h5"

mkdir -p "$OUT_DIR" logs

echo "host=$(hostname) task=$TASK model=$MODEL seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/plm_choice"

# Everything cheap and local is checked before `uv sync` and the model load, so a
# misconfigured array fails in seconds rather than after the environment is built.
# (Both checks come after the cd, so a RELATIVE override is resolved against the
# same directory the Python process will use, not the submit directory.)
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA not found: $FASTA (override with FASTA=... sbatch ...)" >&2
    exit 6
fi

# Deliberately NOT "exists -> exit 0". -f tests existence, not completeness, and
# the batched writer flushes as it goes — so a task killed at the wall clock leaves
# a readable but partial file. Skipping on existence would print "already present",
# exit 0 and mail END: the grid reports complete while one seed's probe trained on
# a fraction of the cohort, which is the fabricated-error-bar failure this branch
# exists to close, just non-zero this time. The writer opens "w-", so the only
# correct recovery is to delete the file and rerun; say so.
if [ -f "$OUT" ]; then
    echo "ERROR: $OUT already exists. It may be a completed arm or a wall-clock" >&2
    echo "       truncation — check its key count against the FASTA before reusing" >&2
    echo "       it. To regenerate: rm '$OUT' && sbatch --array=$TASK ..." >&2
    exit 4
fi

# uv lives in ~/.local/bin, which is added by the login-shell profile. A batch
# job is NOT a login shell, so `uv` resolves only if the submitting environment
# happened to export it (Slurm's default --export=ALL). Relying on that means
# the job dies with "uv: command not found" when submitted from anywhere else.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "ERROR: uv not found on PATH ($PATH)" >&2; exit 3; }

uv sync --locked

# Fail here, on one task, rather than discovering after the whole array drains
# that the grid silently missed an arm (or queued one that argparse will reject).
REGISTRY="$(uv run python -c 'from data_preparation.embeddings.embedding_generation import MODEL_CONFIGS; print("\n".join(sorted(MODEL_CONFIGS)))')"
EXPECTED="$(printf '%s\n' "${MODELS[@]}" | sort)"
if [ "$REGISTRY" != "$EXPECTED" ]; then
    echo "ERROR: this script's MODELS list has drifted from MODEL_CONFIGS." >&2
    diff <(echo "$EXPECTED") <(echo "$REGISTRY") >&2 || true
    echo "       Update MODELS (and --array bounds: $N_MODELS models x $N_SEEDS seeds)." >&2
    exit 5
fi

# --token_budget turns ~481k single-sequence forward passes into length-bucketed
# batches. 16384 suits an 80GB A100; lower it for esm2_3b if a task OOMs (an OOM
# now halves the batch and retries rather than dropping the whole tail to one
# sequence per pass).
#
# Numerics: for the fp32 arms this is exact — batched and unbatched HDF5 agree to
# 4.8e-07. For prot_t5 and prost_t5 it is NOT: those load in fp16, and the
# unbatched path takes its mean in numpy at fp16 while the batched path pools in
# fp32, so the two differ by ~3.7e-04 relative. The batched value is the more
# accurate one, but it means an fp16 arm regenerated here is not bit-comparable
# with a pretrained twin produced by the old path.
#
# --bf16 is deliberately NOT set. It would speed this up further, but it changes
# the vectors, and the pretrained twins these are subtracted from were produced
# in fp32/fp16. Turning it on for the untrained arm alone would reintroduce
# exactly the precision confound this branch just removed. If bf16 is wanted,
# regenerate BOTH arms with it.
uv run python -m data_preparation.embeddings.embedding_generation \
    "$FASTA" "$MODEL" \
    --random_init \
    --random_seed "$SEED" \
    --output_hdf5_file "$OUT" \
    --embedding_type per_protein \
    --token_budget "${TOKEN_BUDGET:-16384}"

echo "wrote $OUT"
