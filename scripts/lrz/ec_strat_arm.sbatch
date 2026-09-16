#!/bin/bash
# C1 EC functional arm over the stratified cohort, one array task per pLM (M-14 / M-15).
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --test-only --array=0-14 scripts/lrz/ec_strat_arm.sbatch
#     sbatch            --array=0-14 scripts/lrz/ec_strat_arm.sbatch          # euclidean (primary)
#     DISTANCE=cosine OUT=~/c1_strat/out_cosine sbatch --array=0-14 scripts/lrz/ec_strat_arm.sbatch
#
# ---------------------------------------------------------------------------
# WHY A STRATIFIED 3,503 AND NOT THE FULL 11,121. Cost is quadratic: the statistic is
# re-evaluated over every induced pair on each of ~5,000 bootstrap/permutation draws, so
# C(n,2) is the multiplier. Measured on an M4 Pro: a full tau_b costs 11.4 s at n=11,121
# versus 1.2 s at n=3,500 -- 15.8 h/cell against 1.6 h/cell, a 10x saving for a CI
# half-width that only widens from ~0.0035 to ~0.0068, far inside the between-arm spread
# this grid has to resolve.
#
# The stratification is not a compromise, it is the point. The full cohort is dominated by
# redundant near-identical sequences, so a random draw inherits that skew and reproduces
# exactly the saturation reviewers R1.4/R1.6 complained about. Strata over EC top-level
# class x length quartile (proportional, floor 8 per cell) guarantee all seven EC classes
# and a flat length profile: 1:806 2:1247 3:788 4:323 5:177 6:118 7:44, quartiles
# 882/873/877/871.
#
# THE COHORT IS PRE-FILTERED TO <=1022 RESIDUES, AND THAT IS LOAD-BEARING. ec_report reads
# RAW h5 keys (analysis_io.load_embeddings_h5) and never applies the load-time cohort
# filter in shared/datasets.py, then raises PopulationError on any frozen id an arm lacks.
# An id of 1023-2000 aa fails clean/esm1b; an id >2000 aa fails ALL FIFTEEN arms, because
# the M-12 cohort cut removed those physically. The exclusion freeze cannot catch the
# second case -- its own `universe` is 540,881, i.e. it is defined after the <=2000 cut.
# So the freeze here was built as `cohort INTERSECT {L <= 1022}` from sprot.fasta.fai.
# Never pass --allow-capped to "fix" a failure: it silently changes the scored population
# per arm, which is the exact defect the cohort freeze exists to prevent.
#
# LABELS ARE PIVOTED, ONE ROW PER PROTEIN. label_adapters.parse_ec appends one record per
# ROW with no grouping, so the long-form ec_annot.tsv (one row per EC) would emit duplicate
# label rows for multi-EC proteins and trip a bare assert that main() does not catch.
# ec_labels.tsv is Entry + "EC number" with ';'-joined ECs, passed via --ec-col.
#
# Validated end to end before submission on 80 ids against the local prostt5.h5:
# tau_b=0.2013 rho=0.2515 CI [0.1465, 0.2893] perm_p=0.0196, exit 0.
# ---------------------------------------------------------------------------
#SBATCH -J c1-ec-strat
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=08:00:00              # MANDATORY: default is 1h; measured ~1.6 h/cell
#SBATCH --cpus-per-task=1            # scipy's kendalltau is single-threaded; 1 CPU per
                                     # task keeps every cell backfillable on a contended
                                     # partition (49 pending / 21 idle CPUs at submission)
#SBATCH --mem=12G                    # ~6 GB measured peak at n=3,503, doubled for slack
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
EMB="$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings_cohort2k"
WORK="$HOME/c1_strat"
DISTANCE="${DISTANCE:-euclidean}"
OUT="${OUT:-$WORK/out}"

ARMS=(ankh_base ankh_large clean esm1b esm2_150m esm2_35m esm2_3b esm2_650m
      esm2_8m esm3_open esmc_300m esmc_600m prott5 prottucker random_1024)
N=${#ARMS[@]}
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
if [ "$IDX" -ge "$N" ]; then echo "index $IDX out of range (0-$((N-1)))"; exit 2; fi
ARM=${ARMS[$IDX]}

cd "$HOME/plm_choice"
mkdir -p "$OUT"
echo "arm=$ARM  distance=$DISTANCE  n=3503  out=$OUT"

# random_1024 is the i.i.d. noise floor and prottucker/clean are task-specific arms; all
# three are scored exactly like the rest -- the floor is what makes the others readable.
srun ./.venv/bin/python -m evaluation.ec_report \
    --plm "$ARM" \
    --emb-h5 "$EMB/$ARM.h5" \
    --freeze "$WORK/ec_strat_freeze.json" \
    --ec-tsv "$WORK/ec_labels.tsv" \
    --ec-col "EC number" \
    --out-dir "$OUT" \
    --distance "$DISTANCE" \
    --statistic tau_b \
    --n-boot 2000 \
    --n-perm 1000 \
    --seed 42

echo "done: $ARM"
