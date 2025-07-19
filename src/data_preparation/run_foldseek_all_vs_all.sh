#!/bin/bash

# Script to perform an all-vs-all structural similarity search using Foldseek
# Supports both PDB/mmCIF directories and Foldcomp databases

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
INPUT_PATH="$1"
OUTPUT_DIR="$2"
THREADS="${3:-8}" # Default to 8 threads if not provided
INPUT_FORMAT="${4:-auto}" # Default to auto-detection

# --- Validate Input ---
if [ -z "$INPUT_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <input_path> <output_dir> [threads] [input_format]"
    echo ""
    echo "Arguments:"
    echo "  input_path    : Path to PDB/mmCIF directory OR foldcomp database"
    echo "  output_dir    : Output directory for results"
    echo "  threads       : Number of threads (default: 8)"
    echo "  input_format  : Input format (auto|pdb|foldcomp) (default: auto)"
    echo ""
    echo "Examples:"
    echo "  # PDB directory"
    echo "  $0 data/interm/2024_new/colabfold/pdb data/interm/2024_new/foldseek 8 pdb"
    echo ""
    echo "  # Foldcomp database"
    echo "  $0 data/interm/sprot_pre2024/compressed/af_test_compressed data/interm/foldseek 8 foldcomp"
    echo ""
    echo "  # Auto-detection (recommended)"
    echo "  $0 data/interm/2024_new/colabfold/pdb data/interm/2024_new/foldseek 8"
    exit 1
fi

if ! command -v foldseek &>/dev/null; then
    echo "Error: foldseek command not found. Please ensure it is installed and in your PATH."
    echo "Installation instructions: https://github.com/deepmind/foldseek"
    exit 1
fi

# --- Auto-detect input format ---
detect_input_format() {
    local input_path="$1"

    # Check if it's a foldcomp database (has .dbtype file)
    if [ -f "${input_path}.dbtype" ]; then
        echo "foldcomp"
        return
    fi

    # Check if it's a directory
    if [ -d "$input_path" ]; then
        # Check for PDB/mmCIF files in directory
        local pdb_count=$(find "$input_path" -name "*.pdb" -o -name "*.pdb.gz" -o -name "*.cif" -o -name "*.cif.gz" | wc -l)
        if [ "$pdb_count" -gt 0 ]; then
            echo "pdb"
            return
        fi

        # Check for foldcomp files in directory
        local fcz_count=$(find "$input_path" -name "*.fcz" | wc -l)
        if [ "$fcz_count" -gt 0 ]; then
            echo "foldcomp"
            return
        fi
    fi

    # Check file extension
    case "$input_path" in
        *.fcz) echo "foldcomp" ;;
        *.pdb|*.pdb.gz|*.cif|*.cif.gz) echo "pdb" ;;
        *) echo "unknown" ;;
    esac
}

if [ "$INPUT_FORMAT" = "auto" ]; then
    DETECTED_FORMAT=$(detect_input_format "$INPUT_PATH")
    if [ "$DETECTED_FORMAT" = "unknown" ]; then
        echo "Error: Could not auto-detect input format for '$INPUT_PATH'."
        echo "Please specify format explicitly: pdb or foldcomp"
        exit 1
    fi
    INPUT_FORMAT="$DETECTED_FORMAT"
    echo "Auto-detected input format: $INPUT_FORMAT"
fi

# --- Validate input based on format ---
if [ "$INPUT_FORMAT" = "foldcomp" ]; then
    # For foldcomp, check if database exists (either single file or directory with .dbtype)
    if [ -f "${INPUT_PATH}.dbtype" ]; then
        echo "Found foldcomp database: $INPUT_PATH"
        STRUCTURE_COUNT=$(wc -l < "${INPUT_PATH}.lookup" 2>/dev/null || echo "unknown")
    elif [ -d "$INPUT_PATH" ]; then
        FCZ_COUNT=$(find "$INPUT_PATH" -name "*.fcz" | wc -l)
        if [ "$FCZ_COUNT" -eq 0 ]; then
            echo "Error: No foldcomp (.fcz) files found in directory '$INPUT_PATH'."
            exit 1
        fi
        STRUCTURE_COUNT="$FCZ_COUNT"
        echo "Found foldcomp directory with $FCZ_COUNT .fcz files"
    else
        echo "Error: Foldcomp input '$INPUT_PATH' not found or invalid."
        echo "Expected either a foldcomp database (with .dbtype file) or directory with .fcz files."
        exit 1
    fi
elif [ "$INPUT_FORMAT" = "pdb" ]; then
    # For PDB, check directory exists and has PDB/mmCIF files
    if [ ! -d "$INPUT_PATH" ]; then
        echo "Error: PDB directory '$INPUT_PATH' does not exist."
        exit 1
    fi

    STRUCTURE_COUNT=$(find "$INPUT_PATH" -name "*.pdb" -o -name "*.pdb.gz" -o -name "*.cif" -o -name "*.cif.gz" | wc -l)
    if [ "$STRUCTURE_COUNT" -eq 0 ]; then
        echo "Error: No PDB/mmCIF files found in directory '$INPUT_PATH'."
        exit 1
    fi
    echo "Found PDB/mmCIF directory with $STRUCTURE_COUNT structure files"
else
    echo "Error: Invalid input format '$INPUT_FORMAT'. Must be 'pdb' or 'foldcomp'."
    exit 1
fi

# --- Directory Setup ---
BASE_NAME=$(basename "$INPUT_PATH")
TMP_DIR="${OUTPUT_DIR}/tmp_${BASE_NAME}"
RESULTS_DIR="${OUTPUT_DIR}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"

DB_NAME="${TMP_DIR}/${BASE_NAME}_db"
RESULT_DB_NAME="${TMP_DIR}/${BASE_NAME}_results_db"
SEARCH_TMP_DIR="${TMP_DIR}/${BASE_NAME}_search_tmp" # Temporary directory for foldseek search
RESULT_TSV="${RESULTS_DIR}/${BASE_NAME}_all_vs_all.tsv"

echo "--- Starting All-vs-All Foldseek Search ---"
echo "Input Path: $INPUT_PATH"
echo "Input Format: $INPUT_FORMAT"
echo "Structure count: $STRUCTURE_COUNT"
echo "Output Directory: $OUTPUT_DIR"
echo "Temporary Directory: $TMP_DIR"
echo "Results TSV: $RESULT_TSV"
echo "Threads: $THREADS"
echo "-------------------------------------------"

# Step 1: Create Foldseek Database
echo "Step 1: Creating Foldseek database..."
if [ ! -f "${DB_NAME}.dbtype" ]; then
    if [ "$INPUT_FORMAT" = "foldcomp" ]; then
        echo "Creating database from foldcomp input with --input-format 5..."
        foldseek createdb "$INPUT_PATH" "$DB_NAME" --input-format 5
    else
        echo "Creating database from PDB/mmCIF input..."
        foldseek createdb "$INPUT_PATH" "$DB_NAME"
    fi
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
        -s 7.5 \
        -e 0.001 \
        -a \
        --threads "$THREADS"
        # --alignment-mode 3 \
        # --max-seqs 1000 \
        # --num-iterations 3 \
        # --e-profile 1e-10 \
        # --exhaustive-search 1
    echo "Search completed. Results database: $RESULT_DB_NAME"
else
    echo "Search results database ${RESULT_DB_NAME} already exists. Skipping search."
fi

# Step 3: Convert Results to Human-Readable Format
echo "Step 3: Converting results to TSV format..."
# Format output: query, target, fident, evalue, qcov, tcov, lddt, rmsd, alntmscore, qlen, tlen
foldseek convertalis "$DB_NAME" "$DB_NAME" "$RESULT_DB_NAME" "$RESULT_TSV" \
    --format-mode 4 \
    --format-output "query,target,fident,evalue,qcov,tcov,alntmscore"
echo "Conversion complete. Results saved to: $RESULT_TSV"

# Step 4: Display summary statistics
echo "--- Summary Statistics ---"
TOTAL_COMPARISONS=$(tail -n +1 "$RESULT_TSV" | wc -l)
echo "Total pairwise comparisons: $TOTAL_COMPARISONS"

# Display first few lines (excluding header if any)
echo "First 5 results:"
head -5 "$RESULT_TSV"

echo "--- All-vs-All Foldseek Search Finished Successfully ---"
echo "Input: $INPUT_PATH ($INPUT_FORMAT format)"
echo "Output file: $RESULT_TSV"
echo "Columns: query, target, fident, evalue, qcov, tcov, alntmscore"

# Clean up temporary directory
# echo "Cleaning up temporary directory: $TMP_DIR"
# rm -rf "$TMP_DIR"

exit 0
