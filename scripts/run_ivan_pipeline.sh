#!/usr/bin/env bash
# --- Ivan infrastructure (2026-03-19) ---
#
# Full Ivan validation pipeline runner for pLM Choice revision.
#
# Chains all Ivan infrastructure scripts in the correct order:
#   1. Download reference data (GO ontology, SIFTS, SCOP, ECOD, EC)
#   2. GO semantic similarity (Wang method) on test pairs, merge into splits
#   3. EC hierarchy distances  — NOT AVAILABLE, see the step 3 block below
#   4. PDB experimental TM-scores on test pairs, merge into splits
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
# Override these to point at your data layout. Defaults follow the project
# convention: data/processed/sprot_pre2024/ with sets/ and embeddings/ subdirs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Root of the processed dataset (contains sets/ and embeddings/ subdirs)
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/processed/sprot_pre2024}"

# Directory with train.parquet, val.parquet, test.parquet
SETS_DIR="${SETS_DIR:-${DATA_DIR}/sets}"

# Directory with per-model .h5 embedding files
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-${DATA_DIR}/embeddings}"

# Pipeline output directory for intermediate files
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/out/ivan_pipeline}"

# Reference data directory (GO ontology, SIFTS, SCOP, ECOD, EC annotations)
REFERENCE_DIR="${REFERENCE_DIR:-${PROJECT_ROOT}/data/reference}"

# FASTA file for embedding generation (random init baselines)
FASTA_FILE="${FASTA_FILE:-${DATA_DIR}/sequences.fasta}"

# GO annotations file (CAFA5 train_terms.tsv or custom TSV)
GO_ANNOTATIONS="${GO_ANNOTATIONS:-${REFERENCE_DIR}/go/cafa5/Train/train_terms.tsv}"

# EC annotations file
EC_ANNOTATIONS="${EC_ANNOTATIONS:-${REFERENCE_DIR}/ec/uniprot_ec_reviewed.tsv}"

# SCOP/ECOD classification parquet for step 7
CLASSIFICATION_PARQUET="${CLASSIFICATION_PARQUET:-${DATA_DIR}/scop_classifications.parquet}"

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
            echo "  DATA_DIR, SETS_DIR, EMBEDDINGS_DIR, OUTPUT_DIR,"
            echo "  REFERENCE_DIR, FASTA_FILE, GO_ANNOTATIONS,"
            echo "  EC_ANNOTATIONS, CLASSIFICATION_PARQUET"
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
        --pairs_parquet "${SETS_DIR}/test.parquet" \
        --output_parquet "${GO_OUTPUT}" \
        --obo_path "${REFERENCE_DIR}/go/go-basic.obo" \
        --aspects MFO BPO CCO

    # 2b: Merge GO columns into train/val/test splits
    echo "  [2b] Merging GO columns into train/val/test..."
    uv run python src/data_preparation/merge_parquet_columns.py \
        --source "${GO_OUTPUT}" \
        --target_dir "${SETS_DIR}" \
        --columns go_wang_mfo go_wang_bpo go_wang_cco
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
        --pairs_parquet "${SETS_DIR}/test.parquet" \
        --output_parquet "${TMSCORE_OUTPUT}" \
        --pdb_cache_dir "${REFERENCE_DIR}/pdb_cache" \
        --sifts_mapping "${REFERENCE_DIR}/sifts/uniprot_pdb.tsv" \
        --max_workers 4

    # 4b: Merge tmscore_exp column into train/val/test splits
    echo "  [4b] Merging tmscore_exp into train/val/test..."
    uv run python src/data_preparation/merge_parquet_columns.py \
        --source "${TMSCORE_OUTPUT}" \
        --target_dir "${SETS_DIR}" \
        --columns tmscore_exp
}

# --------------------------------------------------------------------------- #
#                    STEP 5: BRENDA/HFSP Validation
# --------------------------------------------------------------------------- #

BRENDA_OUTPUT="${OUTPUT_DIR}/hfsp_validation/hfsp_validation_3_5_2_6.json"

step_5_brenda_validation() {
    # Run HFSP validation on beta-lactamases (EC 3.5.2.6)
    echo "  Validating HFSP on beta-lactamases (EC 3.5.2.6)..."
    uv run python src/data_preparation/brenda_hfsp_validation.py \
        --pairs_parquet "${SETS_DIR}/test.parquet" \
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
RANDOM_INIT_ESM2="${EMBEDDINGS_DIR}/random_init_esm2_650m_seed${RANDOM_SEED}.h5"
RANDOM_INIT_PROTT5="${EMBEDDINGS_DIR}/random_init_prot_t5_seed${RANDOM_SEED}.h5"

step_6_random_init() {
    # Check that FASTA file exists
    if [[ ! -f "${FASTA_FILE}" ]]; then
        echo "  ERROR: FASTA file not found: ${FASTA_FILE}" >&2
        echo "  Set FASTA_FILE to the path of your sequences FASTA." >&2
        return 1
    fi

    # 6a: Random-init ESM2-650M
    if [[ -f "${RANDOM_INIT_ESM2}" && "$FORCE" == false ]]; then
        echo "  SKIP: $(basename "${RANDOM_INIT_ESM2}") already exists"
    else
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
    uv run python src/evaluation/classification_eval.py \
        --pairs_parquet "${SETS_DIR}/test.parquet" \
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
echo "  DATA_DIR:               ${DATA_DIR}"
echo "  SETS_DIR:               ${SETS_DIR}"
echo "  EMBEDDINGS_DIR:         ${EMBEDDINGS_DIR}"
echo "  OUTPUT_DIR:             ${OUTPUT_DIR}"
echo "  REFERENCE_DIR:          ${REFERENCE_DIR}"
echo "  FASTA_FILE:             ${FASTA_FILE}"
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
