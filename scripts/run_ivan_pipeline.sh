#!/usr/bin/env bash
# --- Ivan infrastructure (2026-03-19) ---
#
# Full Ivan validation pipeline runner for pLM Choice revision.
#
# Chains all Ivan infrastructure scripts in the correct order:
#   1. Download reference data (GO ontology, SIFTS, SCOP, ECOD, EC)
#   2. GO semantic similarity (Wang method) on test pairs
#   3. EC hierarchy distances  — NOT AVAILABLE, see the step 3 block below
#   4. PDB experimental TM-scores on test pairs
#   5. BRENDA/HFSP validation (produces JSON report, no merge)
#   6. Random-init baselines (esm2_650m and prot_t5)
#   7. Classification evaluation at SCOP/ECOD hierarchy levels
#
# Each step is a function that can be run independently via --step N.
# Steps check for existing output and skip if present (idempotent).
# Use --force to re-run even if output exists.
#
# Usage:
#   ./scripts/run_ivan_pipeline.sh                  # run all steps
#   ./scripts/run_ivan_pipeline.sh --step 2         # run only step 2
#   ./scripts/run_ivan_pipeline.sh --force           # re-run everything
#   ./scripts/run_ivan_pipeline.sh --dry-run         # show what would be done
#   ./scripts/run_ivan_pipeline.sh --step 6 --force  # re-run step 6
#
set -euo pipefail

# --------------------------------------------------------------------------- #
#                         CONFIGURABLE VARIABLES
# --------------------------------------------------------------------------- #
# There used to be a single DATA_DIR here, described as "the project convention:
# data/processed/sprot_pre2024/". That was the only cohort-policy statement
# tracked in git and it was WRONG — and worse, no single value could have been
# right, because DATA_DIR was answering three different questions at once and two
# of them have OPPOSITE correct answers:
#
#   role                                  wants    why
#   ------------------------------------  -------  ---------------------------------
#   annotation READ source (2a/4a/5/7)     FULL     subset test pairs are a strict 10%
#                                                   sample of full test pairs, so one
#                                                   computation covers BOTH cohorts at
#                                                   100%; computing on the subset covers
#                                                   10% and cannot be back-filled
#   merge WRITE target (was 2b/4b)         neither  destructive; see step 2
#   step 6 .h5 destination                 SUBSET   matches scripts/lrz/embed_random_init.sbatch
#
# Cohort sizes (train / val / test pairs):
#   sprot_pre2024        113,186,256 / 16,105,295 / 15,719,249   (FULL)
#   sprot_pre2024_subset  11,318,625 /  1,610,529 /  1,571,924   (uniform 10% row
#                         sample at seed 42 — scripts/create_subset_datasets.py)
#
# The probe trains on the SUBSET: every published cell used it. See
# scripts/lrz/embed_random_init.sbatch and docs/SPECIFICATION.md ("Cohorts").

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# A rename without this guard is strictly WORSE than the old name: an exported
# DATA_DIR would be silently ignored and the caller would get the new defaults
# while believing they had set the cohort.
if [[ -n "${DATA_DIR:-}${SETS_DIR:-}${EMBEDDINGS_DIR:-}" ]]; then
    echo "ERROR: DATA_DIR / SETS_DIR / EMBEDDINGS_DIR are no longer read." >&2
    echo "       One variable cannot answer three questions with two opposite" >&2
    echo "       correct answers. They were split into:" >&2
    echo "         ANNOT_SOURCE_DIR    which cohort's test pairs get annotated (default: FULL)" >&2
    echo "         ANNOT_PAIRS         the exact pairs parquet to read" >&2
    echo "         PROBE_COHORT_DIR    which cohort the probe trains on (default: SUBSET)" >&2
    echo "         RANDOM_INIT_OUT_DIR where step 6 writes its .h5 files" >&2
    echo "       Unset the old name and use the one matching your intent." >&2
    exit 2
fi

# READ-ONLY source for the annotation producers (steps 2a, 4a, 5, 7).
ANNOT_SOURCE_DIR="${ANNOT_SOURCE_DIR:-${PROJECT_ROOT}/data/processed/sprot_pre2024}"
ANNOT_PAIRS="${ANNOT_PAIRS:-${ANNOT_SOURCE_DIR}/sets/test.parquet}"

# The cohort the probe trains on — step 6's outputs are staged against this.
PROBE_COHORT_DIR="${PROBE_COHORT_DIR:-${PROJECT_ROOT}/data/processed/sprot_pre2024_subset}"

# Pipeline output directory, namespaced by the cohort it was computed from.
# Un-namespaced, a FULL-cohort test_with_go.parquet satisfies step 2's run_or_skip
# check for a SUBSET run: the step prints SKIP, and the merge then left-joins the
# FULL cohort's GO table into SUBSET splits on (query, target) — which largely
# SUCCEEDS, because subset pairs are a strict subset. Silently wrong data, exit 0.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/out/ivan_pipeline/$(basename "${ANNOT_SOURCE_DIR}")}"

# Reference data directory (GO ontology, SIFTS, SCOP, ECOD, EC annotations)
REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_ROOT}/data/reference}"

# FASTA file for embedding generation (random init baselines)
# No default: the old ${DATA_DIR}/sequences.fasta names a file that has never existed
# under either cohort and has no producer in this repo. Deliberately NOT defaulted to
# the real data/raw/sprot_2024/sprot.fasta — that would make a 2-of-13-arm laptop demo
# runnable for the first time, duplicating a job scripts/lrz/embed_random_init.sbatch
# already does properly for all 39 arms. The loud failure is a free interlock.
FASTA_FILE="${FASTA_FILE:-}"

# GO annotations file (CAFA5 train_terms.tsv or custom TSV)
GO_ANNOTATIONS="${GO_ANNOTATIONS:-${REFERENCE_DIR}/go/cafa5/Train/train_terms.tsv}"

# EC annotations file
EC_ANNOTATIONS="${EC_ANNOTATIONS:-${REFERENCE_DIR}/ec/uniprot_ec_reviewed.tsv}"

# SCOP/ECOD classification parquet for step 7
# Cohort-INDEPENDENT: classification_eval.py consumes this as a flat protein_id -> class
# map, so it has no cohort. NOTE no producer for it exists in this repo yet, and step 7
# has a second, separate blocker — see the comment above its invocation.
CLASSIFICATION_PARQUET="${CLASSIFICATION_PARQUET:-${REFERENCE_DIR}/scop/scop_classifications.parquet}"

# --------------------------------------------------------------------------- #
#                         CLI ARGUMENT PARSING
# --------------------------------------------------------------------------- #

STEP=""
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--step N] [--force] [--dry-run]"
            echo ""
            echo "Steps:"
            echo "  1  Download reference data (GO, SIFTS, SCOP, ECOD, EC)"
            echo "  2  GO semantic similarity (Wang method)"
            echo "  3  EC hierarchy distances (NOT AVAILABLE — reports and skips)"
            echo "  4  PDB experimental TM-scores"
            echo "  5  BRENDA/HFSP validation"
            echo "  6  Random-init baselines (esm2_650m, prot_t5)"
            echo "  7  Classification evaluation (SCOP/ECOD)"
            echo ""
            echo "Options:"
            echo "  --step N     Run only step N (1-7)"
            echo "  --force      Re-run even if output already exists"
            echo "  --dry-run    Show what would be done without executing"
            echo ""
            echo "Environment variables for custom paths:"
            echo "  ANNOT_SOURCE_DIR, ANNOT_PAIRS, PROBE_COHORT_DIR,"
            echo "  RANDOM_INIT_OUT_DIR, OUTPUT_DIR, REFERENCE_DIR,"
            echo "  FASTA_FILE, GO_ANNOTATIONS, EC_ANNOTATIONS,"
            echo "  CLASSIFICATION_PARQUET"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# --------------------------------------------------------------------------- #
#                         TRACKING
# --------------------------------------------------------------------------- #

STEPS_RAN=()
STEPS_SKIPPED=()

print_merge_hint() {
    # Usage: print_merge_hint <source_parquet> <columns...>
    #
    # The merge is deliberately NOT run automatically. merge_parquet_columns.py
    # rewrites the target splits IN PLACE via atomic_write(..., mode="replace")
    # with NO backup (src/shared/atomic_io.py), and its --splits defaults to
    # train+val+test. Since these producers run on TEST pairs only, the default
    # writes columns that are 100% null into train and val — it warns, then exits
    # 0 with the files already replaced. That was the one path in this script that
    # produced silently wrong data rather than a wasted run.
    local source_file="$1"
    local columns="$2"
    echo ""
    echo "  Computed: ${source_file}"
    echo "  NOT merged into any split — that rewrites train/val/test IN PLACE with no backup."
    echo "  To merge deliberately (note --splits test: the source covers test pairs only):"
    echo ""
    echo "      uv run python src/data_preparation/merge_parquet_columns.py \\"
    echo "          --source \"${source_file}\" \\"
    echo "          --target_dir \"<cohort>/sets\" \\"
    echo "          --columns ${columns} \\"
    echo "          --splits test"
    echo ""
}

run_or_skip() {
    # Usage: run_or_skip <step_number> <output_file> <description> <command...>
    # An empty <output_file> means "no completeness check" — always run the step
    # unless --step filters it out or --dry-run is set.
    local step_num="$1"
    local output_file="$2"
    local description="$3"
    shift 3

    if [[ -n "$STEP" && "$STEP" != "$step_num" ]]; then
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "  Step ${step_num}: ${description}"
    echo "================================================================"

    if [[ -n "$output_file" && -f "$output_file" && "$FORCE" == false ]]; then
        echo "  SKIP: Output already exists: ${output_file}"
        STEPS_SKIPPED+=("${step_num}: ${description}")
        return 0
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "  DRY-RUN: Would execute:"
        echo "    $*"
        STEPS_SKIPPED+=("${step_num}: ${description} (dry-run)")
        return 0
    fi

    echo "  Running..."
    "$@"
    STEPS_RAN+=("${step_num}: ${description}")
}

# --------------------------------------------------------------------------- #
#                    STEP 1: Download Reference Data
# --------------------------------------------------------------------------- #

step_1_download_reference() {
    local download_script="${SCRIPT_DIR}/download_reference_data.sh"
    if [[ ! -f "$download_script" ]]; then
        echo "  ERROR: download_reference_data.sh not found at ${download_script}" >&2
        return 1
    fi

    local args=()
    if [[ "$FORCE" == true ]]; then
        args+=("--force")
    fi

    bash "${download_script}" "${args[@]+"${args[@]}"}"
}

# --------------------------------------------------------------------------- #
#                    STEP 2: GO Semantic Similarity
# --------------------------------------------------------------------------- #

GO_OUTPUT="${OUTPUT_DIR}/test_with_go.parquet"

step_2_go_similarity() {
    # 2a: Compute GO Wang similarity on test pairs
    echo "  [2a] Computing GO semantic similarity on test.parquet..."
    uv run python src/data_preparation/go_semantic_similarity.py \
        --annotations "${GO_ANNOTATIONS}" \
        --pairs_parquet "${ANNOT_PAIRS}" \
        --output_parquet "${GO_OUTPUT}" \
        --obo_path "${REFERENCE_DIR}/go/go-basic.obo" \
        --aspects MFO BPO CCO

    print_merge_hint "${GO_OUTPUT}" "go_wang_mfo go_wang_bpo go_wang_cco"
}

# --------------------------------------------------------------------------- #
#                    STEP 3: EC Hierarchy Distances  (NOT AVAILABLE)
# --------------------------------------------------------------------------- #
#
# `src/data_preparation/ec_hierarchy_distance.py` was deliberately NOT mined in
# from feat/ivan-infrastructure: main already carries evaluation/ec_hierarchy.py
# plus label_adapters.parse_ec, which supersede it (see docs/IVAN_BRANCH_MINING.md,
# "Not taken"). This step still referenced the file, so with `set -e` a plain
# `run_ivan_pipeline.sh` aborted here and steps 4-7 never ran at all.
#
# It now reports and returns cleanly instead of taking the rest of the pipeline
# down with it. Re-enable by writing the pairs->columns producer on top of
# evaluation/ec_hierarchy.py.

EC_OUTPUT="${OUTPUT_DIR}/test_with_ec.parquet"

step_3_ec_distances() {
    echo "  SKIP: EC hierarchy distances are not implemented on this branch." >&2
    echo "        src/data_preparation/ec_hierarchy_distance.py was superseded by" >&2
    echo "        src/evaluation/ec_hierarchy.py, which has no pairs->columns CLI." >&2
    echo "        See docs/IVAN_BRANCH_MINING.md. Continuing with step 4." >&2
    return 0
}

# --------------------------------------------------------------------------- #
#                    STEP 4: PDB Experimental TM-Scores
# --------------------------------------------------------------------------- #

TMSCORE_OUTPUT="${OUTPUT_DIR}/test_with_tmscore_exp.parquet"

step_4_pdb_tmscore() {
    # 4a: Compute experimental TM-scores on test pairs
    echo "  [4a] Computing PDB experimental TM-scores on test.parquet..."
    echo "  (This step downloads PDB structures and runs TMalign — may take hours)"
    uv run python src/data_preparation/pdb_tmscore.py \
        --pairs_parquet "${ANNOT_PAIRS}" \
        --output_parquet "${TMSCORE_OUTPUT}" \
        --pdb_cache_dir "${REFERENCE_DIR}/pdb_cache" \
        --sifts_mapping "${REFERENCE_DIR}/sifts/uniprot_pdb.tsv" \
        --max_workers 4

    print_merge_hint "${TMSCORE_OUTPUT}" "tmscore_exp"
}

# --------------------------------------------------------------------------- #
#                    STEP 5: BRENDA/HFSP Validation
# --------------------------------------------------------------------------- #

BRENDA_OUTPUT="${OUTPUT_DIR}/hfsp_validation/hfsp_validation_3_5_2_6.json"

step_5_brenda_validation() {
    # Run HFSP validation on beta-lactamases (EC 3.5.2.6)
    echo "  Validating HFSP on beta-lactamases (EC 3.5.2.6)..."
    uv run python src/data_preparation/brenda_hfsp_validation.py \
        --pairs_parquet "${ANNOT_PAIRS}" \
        --output_dir "${OUTPUT_DIR}/hfsp_validation" \
        --enzyme_ec 3.5.2.6
}

# --------------------------------------------------------------------------- #
#                    STEP 6: Random-Init Baselines
# --------------------------------------------------------------------------- #

# The seed belongs in the filename: D-6 reports this arm as mean±sd over seeds
# 0/1/2, and a shared path makes the writer skip every already-present protein
# and exit 0, publishing sd = 0.000. This step demonstrates ONE seed; the full
# 13-model x 3-seed grid is scripts/lrz/embed_random_init.sbatch.
RANDOM_SEED="${RANDOM_SEED:-0}"
# Staged OUTSIDE embeddings/, and at the same path scripts/lrz/embed_random_init.sbatch
# uses. run_experiments.py globs "<data_dir>/embeddings/*.h5" with no allowlist, and
# distance_computation.py / all_vs_all_distance_computation.py glob the same directory
# and deliberately do NOT exclude random_init_* — so writing into embeddings/ is an
# implicit "enroll this in the next probe grid and add a dist_ column to the pair
# tables". Promote by moving or symlinking, as a deliberate act.
RANDOM_INIT_OUT_DIR="${RANDOM_INIT_OUT_DIR:-${PROBE_COHORT_DIR}/embeddings_random_init}"
RANDOM_INIT_ESM2="${RANDOM_INIT_OUT_DIR}/random_init_esm2_650m_seed${RANDOM_SEED}.h5"
RANDOM_INIT_PROTT5="${RANDOM_INIT_OUT_DIR}/random_init_prot_t5_seed${RANDOM_SEED}.h5"

step_6_random_init() {
    # Check that FASTA file exists
    if [[ ! -f "${FASTA_FILE}" ]]; then
        echo "  ERROR: FASTA_FILE is unset or missing: '${FASTA_FILE}'" >&2
        echo "  There is deliberately no default. This step is a 2-arm local demo;" >&2
        echo "  the real 13-model x 3-seed grid is:" >&2
        echo "      sbatch --array=0-38 scripts/lrz/embed_random_init.sbatch" >&2
        echo "  For a single local arm:" >&2
        echo "      FASTA_FILE=data/raw/sprot_2024/sprot.fasta \\" >&2
        echo "          ./scripts/run_ivan_pipeline.sh --step 6" >&2
        return 1
    fi

    # 6a: Random-init ESM2-650M
    if [[ -f "${RANDOM_INIT_ESM2}" && "$FORCE" == false ]]; then
        echo "  SKIP: $(basename "${RANDOM_INIT_ESM2}") already exists"
    else
        # Random-init runs open HDF5 with "w-" so a second seed cannot quietly
        # resume into the first seed's file. That makes --force a delete, not an
        # overwrite: without the rm it loads the model, reads the FASTA, and only
        # then dies with FileExistsError.
        rm -f "${RANDOM_INIT_ESM2}"
        echo "  [6a] Generating random-init ESM2-650M embeddings..."
        uv run python src/data_preparation/embeddings/embedding_generation.py \
            "${FASTA_FILE}" esm2_650m \
            --random_init \
            --random_seed "${RANDOM_SEED}" \
            --output_hdf5_file "${RANDOM_INIT_ESM2}" \
            --embedding_type per_protein
    fi

    # 6b: Random-init ProtT5
    if [[ -f "${RANDOM_INIT_PROTT5}" && "$FORCE" == false ]]; then
        echo "  SKIP: $(basename "${RANDOM_INIT_PROTT5}") already exists"
    else
        rm -f "${RANDOM_INIT_PROTT5}"   # "w-" writer: --force means delete first (see 6a)
        echo "  [6b] Generating random-init ProtT5 embeddings..."
        uv run python src/data_preparation/embeddings/embedding_generation.py \
            "${FASTA_FILE}" prot_t5 \
            --random_init \
            --random_seed "${RANDOM_SEED}" \
            --output_hdf5_file "${RANDOM_INIT_PROTT5}" \
            --embedding_type per_protein
    fi
}

# --------------------------------------------------------------------------- #
#                    STEP 7: Classification Evaluation
# --------------------------------------------------------------------------- #

CLASSIF_OUTPUT="${OUTPUT_DIR}/classification_eval/classification_eval_results.parquet"

step_7_classification_eval() {
    if [[ ! -f "${CLASSIFICATION_PARQUET}" ]]; then
        echo "  ERROR: Classification parquet not found: ${CLASSIFICATION_PARQUET}" >&2
        echo "  Create it from SCOP/ECOD data or set CLASSIFICATION_PARQUET." >&2
        return 1
    fi

    echo "  Running classification evaluation at SCOP/ECOD hierarchy levels..."
    # SECOND BLOCKER, independent of the missing classification parquet above: the
    # dist_* columns below are read from test.parquet, but both cohorts' test schema
    # is exactly (query, target, fident, hfsp, alntmscore). The dist_* columns live
    # in train_ext.parquet. classification_eval.py skips all three and then exits 1.
    # Fix by pointing --pairs_parquet at an _ext table, or by running
    # distance_computation.py over ANNOT_PAIRS first.
    uv run python src/evaluation/classification_eval.py \
        --pairs_parquet "${ANNOT_PAIRS}" \
        --classification_parquet "${CLASSIFICATION_PARQUET}" \
        --distance_columns dist_prott5 dist_esm2_650m dist_esm2_3b \
        --hierarchy_columns fold_id sf_id fa_id \
        --output_dir "${OUTPUT_DIR}/classification_eval"
}

# --------------------------------------------------------------------------- #
#                         MAIN EXECUTION
# --------------------------------------------------------------------------- #

echo "============================================================"
echo "  Ivan Validation Pipeline"
echo "  pLM Choice Revision (2026-03-19)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  ANNOT_SOURCE_DIR:       ${ANNOT_SOURCE_DIR}"
echo "  ANNOT_PAIRS:            ${ANNOT_PAIRS}"
echo "  PROBE_COHORT_DIR:       ${PROBE_COHORT_DIR}"
echo "  RANDOM_INIT_OUT_DIR:    ${RANDOM_INIT_OUT_DIR}"
echo "  OUTPUT_DIR:             ${OUTPUT_DIR}"
echo "  REFERENCE_DIR:          ${REFERENCE_DIR}"
echo "  FASTA_FILE:             ${FASTA_FILE:-(unset — required for step 6)}"
echo "  GO_ANNOTATIONS:         ${GO_ANNOTATIONS}"
echo "  EC_ANNOTATIONS:         ${EC_ANNOTATIONS}"
echo "  CLASSIFICATION_PARQUET: ${CLASSIFICATION_PARQUET}"
echo ""
if [[ -n "$STEP" ]]; then
    echo "  Running step: ${STEP} only"
fi
if [[ "$FORCE" == true ]]; then
    echo "  Force mode: ON (re-running even if output exists)"
fi
if [[ "$DRY_RUN" == true ]]; then
    echo "  Dry-run mode: ON (showing commands without executing)"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# --- Run steps ---

# Step 1: Download reference data
# This step has its own idempotency, hence the empty output-file argument.
run_or_skip "1" "" \
    "Download reference data" \
    step_1_download_reference

# Step 2: GO semantic similarity
run_or_skip "2" "${GO_OUTPUT}" \
    "GO semantic similarity (Wang method)" \
    step_2_go_similarity

# Step 3: EC hierarchy distances (reports that it is unavailable, then continues)
run_or_skip "3" "${EC_OUTPUT}" \
    "EC hierarchy distances (NOT AVAILABLE)" \
    step_3_ec_distances

# Step 4: PDB experimental TM-scores
run_or_skip "4" "${TMSCORE_OUTPUT}" \
    "PDB experimental TM-scores" \
    step_4_pdb_tmscore

# Step 5: BRENDA/HFSP validation
run_or_skip "5" "${BRENDA_OUTPUT}" \
    "BRENDA/HFSP validation" \
    step_5_brenda_validation

# Step 6: Random-init baselines
# Uses its own per-file checks, hence the empty output-file argument.
run_or_skip "6" "" \
    "Random-init baselines (esm2_650m, prot_t5)" \
    step_6_random_init

# Step 7: Classification evaluation
run_or_skip "7" "${CLASSIF_OUTPUT}" \
    "Classification evaluation (SCOP/ECOD)" \
    step_7_classification_eval

# --------------------------------------------------------------------------- #
#                         SUMMARY
# --------------------------------------------------------------------------- #

echo ""
echo "============================================================"
echo "  Pipeline Summary"
echo "============================================================"
echo ""

if [[ ${#STEPS_RAN[@]} -gt 0 ]]; then
    echo "  Steps executed:"
    for s in "${STEPS_RAN[@]}"; do
        echo "    [RAN]     ${s}"
    done
else
    echo "  No steps were executed."
fi

if [[ ${#STEPS_SKIPPED[@]} -gt 0 ]]; then
    echo ""
    echo "  Steps skipped (output exists or dry-run):"
    for s in "${STEPS_SKIPPED[@]}"; do
        echo "    [SKIPPED] ${s}"
    done
fi

echo ""
echo "  Output directory: ${OUTPUT_DIR}"
echo ""
echo "Done."
