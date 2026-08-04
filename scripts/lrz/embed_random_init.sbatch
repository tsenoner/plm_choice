#!/bin/bash
# Random-init frozen baseline embeddings (Track B4) on LRZ AI Systems.
#
# Submit as an array over the 12 model x 3 seed grid = 36 tasks (0-35):
#
#     cd ~/plm_choice
#     sbatch --array=0-35 scripts/lrz/embed_random_init.sbatch
#
# Validate without queueing anything first:
#
#     sbatch --test-only --array=0-35 scripts/lrz/embed_random_init.sbatch
#
# Adding a 13th model (esm1b) makes the grid 0-38. The script derives its own
# bounds and refuses an out-of-range index rather than silently mis-pairing.
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
OUT_DIR="${OUT_DIR:-$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_random_init}"

# Keep the HuggingFace cache off the 100 GB backed-up HOME.
export HF_HOME="${HF_HOME:-$DSS/cache/huggingface}"

# The 12 MODEL_CONFIGS keys. esm1b needs a MODEL_CONFIGS entry before it can be
# added here — until then the baseline covers 12 of the paper's 16 arms
# (esm1b, clean, prottucker, random_1024 have no untrained twin) and that gap
# belongs in the manuscript rather than in a footnote nobody writes.
MODELS=(
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

# The probe is trained on the SUBSET splits, so embed the proteins those splits
# reference. Every published cell used sprot_pre2024_subset (11.3M pairs); the
# full splits are 10x larger, and giving the untrained arm 10x the probe budget
# biases the comparison toward it — understating the very quantity being claimed.
FASTA="${FASTA:-$DATA_DIR/raw/sprot_2024/sprot.fasta}"

mkdir -p "$OUT_DIR" logs

echo "host=$(hostname) task=$TASK model=$MODEL seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$HOME/plm_choice"
uv sync --locked

# Explicit per-seed output path. The default would also carry the seed now, but
# being explicit here means a rename upstream cannot silently collapse the grid.
OUT="$OUT_DIR/random_init_${MODEL}_seed${SEED}.h5"
if [ -f "$OUT" ]; then
    echo "already present, nothing to do: $OUT"
    exit 0
fi

uv run python -m data_preparation.embeddings.embedding_generation \
    "$FASTA" "$MODEL" \
    --random_init \
    --random_seed "$SEED" \
    --output_hdf5_file "$OUT" \
    --embedding_type per_protein

echo "wrote $OUT"
