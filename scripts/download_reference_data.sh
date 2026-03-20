#!/usr/bin/env bash
# --- Ivan infrastructure (2026-03-19) ---
#
# Download reference datasets for the Ivan validation pipeline.
#
# Downloads:
#   1. GO ontology (go-basic.obo) from Gene Ontology Consortium
#   2. CAFA5 GO annotations (instructions — requires Kaggle auth)
#   3. SIFTS UniProt-PDB mapping from EBI
#   4. SCOP classification from SCOPe
#   5. ECOD classification from ECOD
#   6. EC annotations from UniProt REST API (reviewed Swiss-Prot only)
#
# All files are stored under data/reference/ with appropriate subdirectories.
# Existing files are skipped (idempotent).
#
# Usage:
#   ./scripts/download_reference_data.sh
#   ./scripts/download_reference_data.sh --force   # re-download everything
#
set -euo pipefail

# --------------------------------------------------------------------------- #
#                           CONFIGURATION
# --------------------------------------------------------------------------- #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REFERENCE_DIR="${PROJECT_ROOT}/data/reference"
GO_DIR="${REFERENCE_DIR}/go"
SIFTS_DIR="${REFERENCE_DIR}/sifts"
SCOP_DIR="${REFERENCE_DIR}/scop"
ECOD_DIR="${REFERENCE_DIR}/ecod"
EC_DIR="${REFERENCE_DIR}/ec"

FORCE=false
if [[ "${1:-}" == "--force" ]]; then
    FORCE=true
fi

# --------------------------------------------------------------------------- #
#                           HELPER FUNCTIONS
# --------------------------------------------------------------------------- #

download_if_missing() {
    local url="$1"
    local output="$2"
    local description="$3"

    if [[ -f "$output" && "$FORCE" == false ]]; then
        echo "  SKIP: ${description} already exists: ${output}"
        return 0
    fi

    echo "  Downloading ${description}..."
    mkdir -p "$(dirname "$output")"

    if curl -fSL --progress-bar -o "${output}" "${url}"; then
        local size
        size=$(du -h "${output}" | cut -f1)
        echo "  OK: ${output} (${size})"
    else
        echo "  FAIL: Could not download ${description} from ${url}" >&2
        rm -f "${output}"
        return 1
    fi
}

download_gz_if_missing() {
    # Download a .gz file, decompress it, and remove the .gz
    local url="$1"
    local output="$2"
    local description="$3"

    if [[ -f "$output" && "$FORCE" == false ]]; then
        echo "  SKIP: ${description} already exists: ${output}"
        return 0
    fi

    echo "  Downloading ${description} (gzipped)..."
    mkdir -p "$(dirname "$output")"

    local gz_output="${output}.gz"
    if curl -fSL --progress-bar -o "${gz_output}" "${url}"; then
        gunzip -f "${gz_output}"
        local size
        size=$(du -h "${output}" | cut -f1)
        echo "  OK: ${output} (${size})"
    else
        echo "  FAIL: Could not download ${description} from ${url}" >&2
        rm -f "${gz_output}" "${output}"
        return 1
    fi
}

# --------------------------------------------------------------------------- #
#                           DOWNLOADS
# --------------------------------------------------------------------------- #

echo "============================================================"
echo "  Ivan Reference Data Downloader"
echo "  Target directory: ${REFERENCE_DIR}"
echo "============================================================"
echo ""

# --- 1. GO ontology ---
echo "[1/6] GO Ontology (go-basic.obo)"
download_if_missing \
    "http://purl.obolibrary.org/obo/go/go-basic.obo" \
    "${GO_DIR}/go-basic.obo" \
    "GO ontology (go-basic.obo)"
echo ""

# --- 2. CAFA5 GO annotations ---
echo "[2/6] CAFA5 GO Annotations"
if [[ -d "${GO_DIR}/cafa5" && "$FORCE" == false ]]; then
    echo "  SKIP: CAFA5 directory already exists: ${GO_DIR}/cafa5"
else
    mkdir -p "${GO_DIR}/cafa5"
    echo "  ================================================================"
    echo "  CAFA5 annotations require Kaggle authentication."
    echo ""
    echo "  To download:"
    echo "    1. Install Kaggle CLI: pip install kaggle"
    echo "    2. Place your kaggle.json in ~/.kaggle/kaggle.json"
    echo "    3. Run:"
    echo "       kaggle competitions download -c cafa-5-protein-function-prediction \\"
    echo "         -p ${GO_DIR}/cafa5/"
    echo "    4. Unzip: unzip ${GO_DIR}/cafa5/*.zip -d ${GO_DIR}/cafa5/"
    echo ""
    echo "  The key file is: Train/train_terms.tsv"
    echo "  (columns: EntryID, term, aspect)"
    echo "  ================================================================"
fi
echo ""

# --- 3. SIFTS UniProt-PDB mapping ---
echo "[3/6] SIFTS UniProt-PDB Mapping"
download_gz_if_missing \
    "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/uniprot_pdb.tsv.gz" \
    "${SIFTS_DIR}/uniprot_pdb.tsv" \
    "SIFTS UniProt-PDB mapping"
echo ""

# --- 4. SCOP classification ---
echo "[4/6] SCOP Classification (SCOPe)"
download_if_missing \
    "https://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt" \
    "${SCOP_DIR}/scop-cla-latest.txt" \
    "SCOP classification (SCOPe 2.08)"
echo ""

# --- 5. ECOD classification ---
echo "[5/6] ECOD Classification"
download_if_missing \
    "http://prodata.swmed.edu/ecod/distributions/ecod.latest.domains.txt" \
    "${ECOD_DIR}/ecod.latest.domains.txt" \
    "ECOD domain classification"
echo ""

# --- 6. EC annotations from UniProt REST API ---
echo "[6/6] EC Annotations from UniProt (reviewed Swiss-Prot)"
EC_OUTPUT="${EC_DIR}/uniprot_ec_reviewed.tsv"
if [[ -f "$EC_OUTPUT" && "$FORCE" == false ]]; then
    echo "  SKIP: EC annotations already exist: ${EC_OUTPUT}"
else
    mkdir -p "${EC_DIR}"
    echo "  Querying UniProt REST API for reviewed entries with EC numbers..."
    echo "  (This may take a few minutes for ~250K entries)"

    # UniProt REST API: fetch accession + ec for all reviewed entries with EC
    # Paginated via Link header; we use a simple curl loop.
    UNIPROT_URL="https://rest.uniprot.org/uniprotkb/search?query=(ec:*)%20AND%20(reviewed:true)&format=tsv&fields=accession,ec&size=500"

    # Paginated download — UniProt REST API returns a Link header with the
    # next-page URL.  We follow it until there's no more data.
    TEMP_FILE=$(mktemp)
    NEXT_URL="${UNIPROT_URL}"
    PAGE=0

    # First page — includes the TSV header line
    HTTP_CODE=$(curl -s -w "%{http_code}" -o "${TEMP_FILE}" -D - \
        -H "User-Agent: plm_choice/1.0 (reference download)" \
        "${NEXT_URL}" 2>/dev/null | head -1 | tr -d '\r\n' | tail -c 3)
    # Proper approach: capture headers separately
    HEADER_FILE=$(mktemp)
    HTTP_CODE=$(curl -s -w "%{http_code}" -o "${TEMP_FILE}" \
        -D "${HEADER_FILE}" \
        -H "User-Agent: plm_choice/1.0 (reference download)" \
        "${NEXT_URL}")

    if [[ "$HTTP_CODE" != "200" ]]; then
        echo "  FAIL: UniProt API returned HTTP ${HTTP_CODE}" >&2
        rm -f "${TEMP_FILE}" "${HEADER_FILE}"
    else
        cp "${TEMP_FILE}" "${EC_OUTPUT}"
        PAGE=1
        echo "  Page ${PAGE} downloaded..."

        # Follow pagination via Link header
        while true; do
            NEXT_URL=$(grep -i '^Link:' "${HEADER_FILE}" | sed 's/.*<\(.*\)>.*/\1/' | tr -d '\r\n')
            if [[ -z "$NEXT_URL" ]]; then
                break
            fi

            rm -f "${HEADER_FILE}"
            HEADER_FILE=$(mktemp)
            HTTP_CODE=$(curl -s -w "%{http_code}" -o "${TEMP_FILE}" \
                -D "${HEADER_FILE}" \
                -H "User-Agent: plm_choice/1.0 (reference download)" \
                "${NEXT_URL}")

            if [[ "$HTTP_CODE" != "200" ]]; then
                echo "  WARN: Page $((PAGE+1)) returned HTTP ${HTTP_CODE}, stopping pagination." >&2
                break
            fi

            # Append data lines (skip header if UniProt repeats it)
            tail -n +2 "${TEMP_FILE}" >> "${EC_OUTPUT}"
            PAGE=$((PAGE + 1))
            echo "  Page ${PAGE} downloaded..."
        done

        rm -f "${TEMP_FILE}" "${HEADER_FILE}"

        local_lines=$(wc -l < "${EC_OUTPUT}" | tr -d ' ')
        local_size=$(du -h "${EC_OUTPUT}" | cut -f1)
        echo "  OK: ${EC_OUTPUT} (${local_lines} lines, ${local_size}, ${PAGE} pages)"
    fi
fi
echo ""

# --------------------------------------------------------------------------- #
#                           SUMMARY
# --------------------------------------------------------------------------- #

echo "============================================================"
echo "  Download Summary"
echo "============================================================"
echo ""

for f in \
    "${GO_DIR}/go-basic.obo" \
    "${SIFTS_DIR}/uniprot_pdb.tsv" \
    "${SCOP_DIR}/scop-cla-latest.txt" \
    "${ECOD_DIR}/ecod.latest.domains.txt" \
    "${EC_DIR}/uniprot_ec_reviewed.tsv"
do
    if [[ -f "$f" ]]; then
        size=$(du -h "$f" | cut -f1)
        printf "  %-50s  %s\n" "$(basename "$f")" "${size}"
    else
        printf "  %-50s  MISSING\n" "$(basename "$f")"
    fi
done

echo ""
echo "Reference data directory: ${REFERENCE_DIR}"
echo "Done."
