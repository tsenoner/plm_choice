#!/usr/bin/env bash
# extract_uniref.sh  –  stream-extract one UniRef50 cluster
# Usage:  ./extract_uniref.sh UniRef50_A0A091DBV8 [/path/to/uniref50.xml.gz]

set -euo pipefail

# ────── 1.  parse arguments ────────────────────────────────────────────────
[[ $# -lt 1 || $# -gt 2 ]] && {
    echo "Usage: $0 <UniRef_ID> [uniref50.xml.gz]" >&2
    exit 1
}

ID="$1"
FILE="${2:-uniref50.xml.gz}" # default if second arg omitted

# ────── 2.  choose gzcat / zcat, depending on OS ──────────────────────────
if command -v gzcat >/dev/null 2>&1; then
    CAT="gzcat" # macOS / *BSD
elif command -v zcat >/dev/null 2>&1; then
    CAT="zcat" # Linux
else
    echo "Error: need gzcat or zcat on PATH" >&2
    exit 2
fi

# ────── 3.  compute output path (same dir as the .gz) ─────────────────────
DIR="$(dirname "$FILE")"
OUT="${DIR}/${ID}.xml"

# If you’d rather refuse to overwrite an existing file, uncomment:
# [[ -e "$OUT" ]] && { echo "Refusing to overwrite $OUT" >&2; exit 3; }

echo "Extracting $ID  →  $OUT"

# ────── 4.  stream-extract and stop immediately after </entry> ────────────
"$CAT" "$FILE" | awk '
    /<entry id=\"'"${ID}"'\"/,/<\/entry>/ {
        print
        if (/<\/entry>/) exit
    }
' >"$OUT" || [[ $? -eq 141 ]]

echo "Done."
