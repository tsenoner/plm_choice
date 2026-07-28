#!/usr/bin/env bash
#
# reproduce.sh — check out this repo and get from nothing to the figures.
#
# The full pipeline is not a single push-button run: embedding 542k proteins across
# 15 pLMs is GPU-weeks, which is exactly why the embeddings are archived on Zenodo.
# This script does the parts that are cheap and deterministic, and prints the exact
# command for each part that is not, rather than pretending to run it.
#
#   ./scripts/reproduce.sh              # verify the environment + run the test suite
#   ./scripts/reproduce.sh --with-data  # additionally fetch the Zenodo deposit (~31 GB)
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Zenodo CONCEPT DOI — always resolves to the newest version of the deposit.
ZENODO_CONCEPT_DOI="10.5281/zenodo.17469267"
ZENODO_API="https://zenodo.org/api/records/17469267"
DATA_DIR="${DATA_DIR:-data/zenodo}"

WITH_DATA=0
[[ "${1:-}" == "--with-data" ]] && WITH_DATA=1

step()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note()  { printf '   %s\n' "$*"; }
manual(){ printf '   \033[2m$ %s\033[0m\n' "$*"; }

# ── 1. Environment ────────────────────────────────────────────────────────────
step "1/4  Environment"
command -v uv >/dev/null || { echo "uv not found — https://docs.astral.sh/uv/"; exit 1; }
uv lock --check
uv sync --locked
uv run plm doctor || true   # advisory: missing bulk data is expected on a fresh clone

# ── 2. Tests ──────────────────────────────────────────────────────────────────
# The suite is the reproducibility claim that costs nothing to check: it pins the
# statistical machinery (bootstrap CIs, vertex-BCa, the barrier specs) that every
# reported number flows through.
step "2/4  Test suite"
uv run pytest -q -m "not slow"
note "Statistical coverage simulations are excluded above; run them with:"
manual "uv run pytest -q -m slow"

# ── 3. Data ───────────────────────────────────────────────────────────────────
step "3/4  Data"
if [[ "$WITH_DATA" -eq 1 ]]; then
  mkdir -p "$DATA_DIR"
  note "Fetching the Zenodo deposit into $DATA_DIR (concept DOI $ZENODO_CONCEPT_DOI)"
  # Resolve the file list from the API so this keeps working across new versions.
  uv run python - "$ZENODO_API" "$DATA_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path

api, dest = sys.argv[1], Path(sys.argv[2])
with urllib.request.urlopen(api) as fh:
    record = json.load(fh)
print(f"record {record['doi']} — {len(record['files'])} files")
manifest = dest / "manifest.txt"
with manifest.open("w") as out:
    for entry in sorted(record["files"], key=lambda f: f["key"]):
        out.write(f"{entry['checksum']}  {entry['key']}  {entry['size']}\n")
        print(f"  {entry['key']:<28} {entry['size'] / 2**30:7.2f} GB  {entry['checksum']}")
print(f"\nmanifest written to {manifest}")
print("download with:  wget -c -i <(python -c \"...links...\")  or the Zenodo web UI")
PY
  note "Verify after downloading:"
  manual "cd $DATA_DIR && md5sum -c <(sed 's/md5://' manifest.txt | awk '{print \$1\"  \"\$2}')"
else
  note "Skipped. Re-run with --with-data to fetch the deposit (~31 GB), or:"
  manual "open https://doi.org/$ZENODO_CONCEPT_DOI"
fi

# ── 4. The analysis DAG ───────────────────────────────────────────────────────
step "4/4  Analysis pipeline"
uv run plm stages
cat <<'EOF'

   The stages above need the bulk inputs. Once DATA_DIR holds the deposit, a
   minimal end-to-end pass over one pLM and one target looks like:

     uv run plm data merge   --help          # build the pair table
     uv run plm data split   --help          # train/val/test parquet (record sha256)
     uv run plm train sweep  --csv_dir <splits> --model_types euclidean fnn \
                             --evaluate_after_train
     uv run plm figures summary --results_dir models/<dataset> --output out/plots

   Each command's own --help lists its real options; nothing is re-declared by the
   CLI, so the help you see is the module's.

   Determinism: seeds are set per run and each split's sha256 is recorded. Set
   PYTHONHASHSEED=0 and MPLBACKEND=Agg (as CI does) for byte-stable figures.

EOF

step "Done"
note "Environment verified and the test suite passed."
