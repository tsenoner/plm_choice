#!/bin/bash

# This script generates embeddings for a given FASTA file using all
# supported models in the unified_embedder.py script.
# Each model's embeddings will be saved to a separate HDF5 file named
# according to the pattern: [fasta_filename_stem]_[model_key]_embeddings.h5

# --- Configuration ---
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_fasta_file>"
    exit 1
fi
FASTA_FILE="$1"

if [ ! -f "${FASTA_FILE}" ]; then
    echo "Error: FASTA file not found at ${FASTA_FILE}"
    exit 1
fi

# Path to the unified embedder script
EMBEDDER_SCRIPT_REL_PATH="src/data_preparation/embeddings/embedding_generation.py"
# Attempt to find the script relative to this script's location or assuming common project structure
SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="${SCRIPT_DIR}/../../.."
EMBEDDER_SCRIPT="${PROJECT_ROOT}/${EMBEDDER_SCRIPT_REL_PATH}"

if [ ! -f "${EMBEDDER_SCRIPT}" ]; then
    echo "Error: unified_embedder.py script not found at ${EMBEDDER_SCRIPT}"
    echo "Please ensure EMBEDDER_SCRIPT_REL_PATH is set correctly or the script is in the expected location."
    exit 1
fi

# Optional: Path to your Hugging Face token.
HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"

# --- Model Keys (extracted from unified_embedder.py MODEL_CONFIGS) ---
MODEL_KEYS=(
    "esm2_8m"
    "esm2_35m"
    "esm2_150m"
    "esm2_650m"
    "esm2_3b"
    "esm3_open"
    "esmc_300m"
    "esmc_600m"
    "ankh_base"
    "ankh_large"
    "prot_t5_xl"
    "prost_t5"
)

# --- Main Loop ---
echo "Starting embedding generation for: ${FASTA_FILE}"
echo "Output HDF5 files will be generated in the same directory as the FASTA file, suffixed by model name."
echo "---"

for model_key in "${MODEL_KEYS[@]}"; do
    echo "Processing model: ${model_key}"

    # Construct the command - unified_embedder.py now handles output file naming
    CMD=(
        "python"
        "${EMBEDDER_SCRIPT}"
        "${FASTA_FILE}"
        "${model_key}"
        # --output_hdf5_file is now optional in unified_embedder.py
        "--embedding_type" "per_protein" # Or "per_residue"
        # Add --max_seq_len if needed, e.g., "--max_seq_len" "1000"
    )

    # Add token path if the file exists
    if [ -f "${HF_TOKEN_PATH}" ]; then
        CMD+=("--token_path" "${HF_TOKEN_PATH}")
    fi

    echo "Running command: ${CMD[*]}"

    # Execute the command
    "${CMD[@]}"

    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: Embedding generation failed for model ${model_key} with exit code ${EXIT_CODE}."
        # Decide if you want to stop on error or continue with other models
        # exit ${EXIT_CODE} # Uncomment to stop on first error
    else
        echo "Successfully generated embeddings for ${model_key}."
    fi
    echo "---"
done

echo "All models processed."
