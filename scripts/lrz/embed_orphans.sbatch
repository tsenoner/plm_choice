#!/bin/bash
# C8 orphan arm, step 1: embed the 11,444 metagenomic "orphan" proteins with every pLM the
# repo can build, one array task per model.
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --array=0-11%4 scripts/lrz/embed_orphans.sbatch
#
# ---------------------------------------------------------------------------
# THE SET. Prabakaran & Bromberg, Bioinformatics 41(2):btaf035 (2025), figshare
# 10.6084/m9.figshare.c.6737127: 11,444 ESM-Atlas/MGnify proteins with <30% identity to UniRef100,
# <=400 aa, CD-HIT 90, plus 309,549 structurally alignable pairs of which 6,219 are "siblings"
# (TM >= 0.7 AND Siamese-NN >= 0.98). This IS the "MGnify ~11k" set the plan names, and
# evaluation/orphan_io.py already parses its pair table.
#
# WHAT THIS ARM CAN AND CANNOT CLAIM (write these, do not discover them again):
#   * "orphan" here means <30% identity, NOT dark matter -- HMMER/Pfam annotates 8,661 of 11,444.
#   * ESM-C trains on MGnify, ESM3 lists MGnify90 2023_02 and ESMAtlas, and ProtT5-XL-U50 descends
#     from BFD, which includes metagenomic assemblies. Those three arms are contaminated w.r.t.
#     this set. Uncontaminated: ESM-1b, ESM-2, Ankh (and CLEAN/ProtTucker, built on them).
#   * The sibling labels come from ESMFold structures (ESM-2 3B trunk) plus a Siamese net on DNA
#     embeddings, so the ESM family is partly judged against its own predictions.
#   * Non-sibling pairs are UNLABELLED, not negative, so plain AUROC flatters; report the paper's
#     PU-aware statistic beside it.
#
# 12 keys, not 15: MODEL_CONFIGS has no clean or prottucker (they are projection heads on esm1b
# and prott5 -- build them from these outputs afterwards, as was done for New2024), prost_t5 is
# out of the study entirely, and random_1024 comes from random_embeddings.py (UNSEEDED -- record
# the file hash if you make one).
#
# Deliberately NOT batched (--token_budget stays 0): the Swiss-Prot cohort these results will be
# read beside was embedded one sequence at a time, and 11,444 sequences of <=400 aa is cheap.
# ---------------------------------------------------------------------------
#SBATCH -J embed-orphan
#SBATCH -p lrz-hgx-a100-80x4,lrz-hgx-h100-94x4,lrz-dgx-a100-80x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1                 # MANDATORY: without it the job sits PD forever
#SBATCH --time=06:00:00              # MANDATORY: default is 1h, QOS caps at 2 days
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
WORK="$DSS/orphan_2026"
FASTA="$WORK/orphan_sequences_11444.fasta"
OUT="$WORK/embeddings"

MODELS=(ankh_base ankh_large esm1b esm2_150m esm2_35m esm2_3b esm2_650m esm2_8m
        esm3_open esmc_300m esmc_600m prot_t5)
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
[ "$IDX" -lt "${#MODELS[@]}" ] || { echo "index $IDX out of range"; exit 2; }
MODEL=${MODELS[$IDX]}

cd "$HOME/plm_choice"
mkdir -p "$OUT"
[ -f "$FASTA" ] || { echo "missing $FASTA"; exit 2; }

# prot_t5 is called prott5 everywhere else in this project; keep the arm names aligned so the
# orphan table can be joined to the Swiss-Prot one without a rename step.
ARM=$MODEL; [ "$MODEL" = prot_t5 ] && ARM=prott5
DEST="$OUT/$ARM.h5"
if [ -f "$DEST" ]; then echo "$ARM already embedded"; exit 0; fi

echo "model=$MODEL arm=$ARM host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

srun ./.venv/bin/python -m data_preparation.embeddings.embedding_generation \
    "$FASTA" "$MODEL" \
    --output_hdf5_file "$DEST.part" \
    --embedding_type per_protein \
    --max_seq_len 2000 \
    --expect_sequences 11444

./.venv/bin/python - "$DEST.part" "$ARM" <<'PYEOF'
import sys, h5py
path, arm = sys.argv[1], sys.argv[2]
with h5py.File(path) as h:
    keys = list(h)
    dim = h[keys[0]].shape
# esm1b clamps at 1022 tokens, so it is allowed to hold fewer; everything else must be complete.
low = 10000 if arm in ("esm1b",) else 11444
if len(keys) < low:
    sys.exit(f"{arm}: only {len(keys)} of 11444 embedded -- refusing to publish a short arm")
print(f"{arm}: {len(keys)} keys, shape {dim}")
PYEOF

mv "$DEST.part" "$DEST"
echo "done: $ARM"
