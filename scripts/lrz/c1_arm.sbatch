#!/bin/bash
# C1/C2 functional + retrieval arms over a stratified cohort, one array task per pLM.
#
#   ARM_KIND=ec  FREEZE=~/c1_strat/ec_strat_v2_freeze.json LABELS=~/c1_strat/ec_labels_v2.tsv \
#     OUT=~/c1_strat/v2_euclidean DISTANCE=euclidean sbatch --array=0-14 c1_arm.sbatch
#   ARM_KIND=fp  FREEZE=~/c1_strat/fp_freeze.json  LABELS=~/c1_strat/cath_labels.tsv \
#     OUT=~/c1_strat/fp_euclidean  DISTANCE=euclidean sbatch --array=0-14 c1_arm.sbatch
#
# Reads the FULL cohort2k h5 per arm (not the 3,503-protein laptop slices): the CATH-capped
# v2 cohort adds 4,740 proteins the slices do not contain.
#
# The cohort is pre-filtered to <=1022 residues. ec_report/recall_fp_report read RAW h5 keys
# and never apply the load-time cohort filter, then raise PopulationError on any id an arm
# lacks -- an id of 1023-2000 aa fails clean/esm1b, an id >2000 aa fails ALL FIFTEEN, since
# the M-12 cut removed those physically. Never pass --allow-capped to get past a failure.
#SBATCH -J c1-arm
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
set -euo pipefail
unset VIRTUAL_ENV
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
EMB="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
ARMS=(ankh_base ankh_large clean esm1b esm2_150m esm2_35m esm2_3b esm2_650m esm2_8m
      esm3_open esmc_300m esmc_600m prott5 prottucker random_1024)
IDX=${SLURM_ARRAY_TASK_ID:?}; ARM=${ARMS[$IDX]}
cd "$HOME/plm_choice"; mkdir -p "$OUT"
echo "kind=$ARM_KIND arm=$ARM distance=$DISTANCE out=$OUT"
if [ "$ARM_KIND" = "ec" ]; then
  srun ./.venv/bin/python -m evaluation.ec_report --plm "$ARM" --emb-h5 "$EMB/$ARM.h5" \
    --freeze "$FREEZE" --ec-tsv "$LABELS" --ec-col "EC number" --out-dir "$OUT" \
    --distance "$DISTANCE" --statistic tau_b --n-boot 2000 --n-perm 1000 --seed 42
else
  srun ./.venv/bin/python -m evaluation.recall_fp_report --plm "$ARM" --emb-h5 "$EMB/$ARM.h5" \
    --cath-tsv "$LABELS" --freeze "$FREEZE" --out-dir "$OUT" \
    --distance "$DISTANCE" --n-boot 2000 --seed 42
fi
echo "done: $ARM"
