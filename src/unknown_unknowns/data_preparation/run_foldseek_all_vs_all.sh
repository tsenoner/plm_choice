#!/bin/bash

# Script to perform an all-vs-all structural similarity search using Foldseek

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PDB_DIR="$1"
OUTPUT_DIR="$2"
THREADS="${3:-8}" # Default to 8 threads if not provided

# --- Validate Input ---
if [ -z "$PDB_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <pdb_directory> <output_dir> [threads]"
    echo "Example: $0 data/interm/2024_new/colabfold/pdb data/interm/2024_new/foldseek 8"
    exit 1
fi

if ! command -v foldseek &>/dev/null; then
    echo "Error: foldseek command not found. Please ensure it is installed and in your PATH."
    echo "Installation instructions: https://github.com/deepmind/foldseek"
    exit 1
fi

if [ ! -d "$PDB_DIR" ]; then
    echo "Error: PDB directory '$PDB_DIR' does not exist."
    exit 1
fi

# Count PDB files to ensure we have input
PDB_COUNT=$(find "$PDB_DIR" -name "*.pdb" | wc -l)
if [ "$PDB_COUNT" -eq 0 ]; then
    echo "Error: No PDB files found in directory '$PDB_DIR'."
    exit 1
fi

# --- Directory Setup ---
BASE_NAME=$(basename "$PDB_DIR")
TMP_DIR="${OUTPUT_DIR}/tmp_${BASE_NAME}"
RESULTS_DIR="${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

DB_NAME="${TMP_DIR}/${BASE_NAME}_db"
RESULT_DB_NAME="${TMP_DIR}/${BASE_NAME}_results_db"
SEARCH_TMP_DIR="${TMP_DIR}/${BASE_NAME}_search_tmp" # Temporary directory for foldseek search
RESULT_TSV="${RESULTS_DIR}/${BASE_NAME}_all_vs_all.tsv"

echo "--- Starting All-vs-All Foldseek Search ---"
echo "Input PDB Directory: $PDB_DIR"
echo "PDB files found: $PDB_COUNT"
echo "Output Directory: $OUTPUT_DIR"
echo "Temporary Directory: $TMP_DIR"
echo "Results TSV: $RESULT_TSV"
echo "Threads: $THREADS"
echo "-------------------------------------------"

# Step 1: Create Foldseek Database
echo "Step 1: Creating Foldseek database..."
if [ ! -f "${DB_NAME}.dbtype" ]; then
    foldseek createdb "$PDB_DIR" "$DB_NAME"
    echo "Database created: $DB_NAME"
else
    echo "Database ${DB_NAME} already exists. Skipping creation."
fi

# Step 2: Run All-Against-All Structural Search
echo "Step 2: Running all-against-all structural search..."
if [ ! -f "${RESULT_DB_NAME}.dbtype" ]; then
    # Create tmp directory for search if it doesn't exist
    mkdir -p "$SEARCH_TMP_DIR"

    # Run foldseek search with appropriate parameters for structural similarity
    foldseek search "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$SEARCH_TMP_DIR" \
        -s 9 \
        --exhaustive-search 1 \
        -e 0.001 \
        --max-seqs 10000 \
        -a \
        --threads "$THREADS"
    echo "Search completed. Results database: $RESULT_DB_NAME"
else
    echo "Search results database ${RESULT_DB_NAME} already exists. Skipping search."
fi

# Step 3: Convert Results to Human-Readable Format
echo "Step 3: Converting results to TSV format..."
# Format output: query, target, fident, evalue, qcov, tcov, lddt, rmsd, alntmscore, qlen, tlen
foldseek convertalis "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$RESULT_TSV" \
    --format-mode 4 \
    --format-output "query,target,fident,alnlen,mismatch,gapopen,nident,evalue,qcov,tcov,lddt,rmsd,alntmscore,qlen,tlen"
echo "Conversion complete. Results saved to: $RESULT_TSV"

# Step 4: Display summary statistics
echo "--- Summary Statistics ---"
TOTAL_COMPARISONS=$(tail -n +1 "$RESULT_TSV" | wc -l)
echo "Total pairwise comparisons: $TOTAL_COMPARISONS"

# Display first few lines (excluding header if any)
echo "First 5 results:"
head -5 "$RESULT_TSV"

echo "--- All-vs-All Foldseek Search Finished Successfully ---"
echo "Output file: $RESULT_TSV"
echo "Columns: query, target, fident, evalue, qcov, tcov, lddt, rmsd, alntmscore, qlen, tlen"

# Clean up temporary directory
# Comment out if you want to inspect intermediate files
# echo "Cleaning up temporary directory: $TMP_DIR"
# rm -rf "$TMP_DIR"

exit 0
