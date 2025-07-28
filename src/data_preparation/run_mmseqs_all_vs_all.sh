#!/bin/bash

# Script to perform an all-vs-all sequence similarity search using MMseqs2

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
FASTA_FILE="$1"
OUTPUT_DIR="$2"
THREADS="${3:-8}" # Default to 8 threads if not provided

# --- Validate Input ---
if [ -z "$FASTA_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <fasta_file> <output_dir> [threads]"
    exit 1
fi

if ! command -v mmseqs &>/dev/null; then
    echo "Error: mmseqs2 command not found. Please ensure it is installed and in your PATH."
    exit 1
fi

# --- Directory Setup ---
BASE_NAME=$(basename "${FASTA_FILE%.*}")
TMP_DIR="${OUTPUT_DIR}/tmp_${BASE_NAME}"
RESULTS_DIR="${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

DB_NAME="${TMP_DIR}/${BASE_NAME}_db"
RESULT_DB_NAME="${TMP_DIR}/${BASE_NAME}_results_db"
SEARCH_TMP_DIR="${TMP_DIR}/${BASE_NAME}_search_tmp" # Temporary directory for mmseqs search
RESULT_TSV="${RESULTS_DIR}/${BASE_NAME}_all_vs_all.tsv"

echo "--- Starting All-vs-All MMseqs2 Search ---"
echo "Input FASTA: $FASTA_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Temporary Directory: $TMP_DIR"
echo "Results TSV: $RESULT_TSV"
echo "Threads: $THREADS"
echo "-------------------------------------------"

# Step 1: Create MMseqs2 Database
echo "Step 1: Creating MMseqs2 database..."
if [ ! -f "${DB_NAME}.dbtype" ]; then
    mmseqs createdb "$FASTA_FILE" "$DB_NAME"
    echo "Database created: $DB_NAME"
else
    echo "Database ${DB_NAME} already exists. Skipping creation."
fi

# Step 2: Run All-Against-All Search
echo "Step 2: Running all-against-all search..."
if [ ! -f "${RESULT_DB_NAME}.dbtype" ]; then
    mmseqs search "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$SEARCH_TMP_DIR" \
        -s 7.5 \
        -e 0.001 \
        -a \
        --alignment-mode 3 \
        --max-seqs 1000 \
        --num-iterations 3 \
        --e-profile 1e-10 \
        --threads "$THREADS"
        # --exhaustive-search 1
    echo "Search completed. Results database: $RESULT_DB_NAME"
else
    echo "Search results database ${RESULT_DB_NAME} already exists. Skipping search."
fi

# Step 3: Convert Results to Human-Readable Format
echo "Step 3: Converting results to TSV format..."
mmseqs convertalis "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$RESULT_TSV" --format-mode 4 --format-output query,target,fident,alnlen,mismatch,gapopen,nident,qcov,tcov,evalue
# mmseqs convertalis "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$RESULT_TSV" --format-mode 4 --format-output query,target,fident,alnlen,mismatch,gapopen,nident,qaln,taln,qcov,tcov,evalue
echo "Conversion complete. Results saved to: $RESULT_TSV"

echo "--- All-vs-All MMseqs2 Search Finished Successfully ---"
echo "Output file: $RESULT_TSV"
echo "Columns: query, target, fident, alnlen, nident, mismatch, qcov, tcov, evalue"

# Clean up temporary directory
# echo "Cleaning up temporary directory: $TMP_DIR"
# rm -rf "$TMP_DIR"

exit 0
