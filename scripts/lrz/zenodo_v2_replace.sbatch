#!/bin/bash
# Replace files in the Zenodo v2 draft with the cleaned archives from $STAGE/clean/
# (zenodo_v2_scrub.sbatch), each under its published name. Zenodo cannot overwrite a draft
# file, so the draft's copy is deleted and the clean one uploaded, one file at a time (the
# record may not exceed 50 GB); Zenodo's md5 of each upload must equal the local file's.
#
#     sbatch scripts/lrz/zenodo_v2_replace.sbatch          # from the repo root on the cluster
#
# Safe to re-run: a draft file that already has the clean md5 is left alone. Refuses to start
# while a zenodo-rename job is queued or running, because that job would put back the old
# archives. When done it retires .v2_expected_md5.tsv, whose md5s describe the old archives,
# so a later run of zenodo_v2_rename.sbatch stops instead of undoing this. Nothing here publishes.
#
#SBATCH -J zenodo-replace
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

DEPOSIT_ID="${DEPOSIT_ID:-22935411}"
API="https://zenodo.org/api"
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
STAGE="$DSS/zenodo_stage"
CLEAN="$STAGE/clean"
LIMIT=50000000000
TOKEN_FILE="${TOKEN_FILE:-$HOME/.zenodo_token}"

if squeue -u "$USER" -h -o '%j' | grep -qx zenodo-rename; then
    echo "a zenodo-rename job is still queued or running; replace after it finishes"; exit 1
fi
[ -r "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE"; exit 2; }
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
# The token reaches curl through a header file, never argv (visible to every user via ps).
printf 'Authorization: Bearer %s\n' "$(tr -d ' \n' < "$TOKEN_FILE")" > "$WORK/auth"
api () { curl -sS --fail-with-body -H @"$WORK/auth" "$@"; }

api "$API/deposit/depositions/$DEPOSIT_ID" > "$WORK/dep.json"
read -r STATE BUCKET < <(python3 -c '
import json, sys; d = json.load(open(sys.argv[1])); print(d["state"], d["links"]["bucket"])' "$WORK/dep.json")
[ "$STATE" != "done" ] || { echo "deposit $DEPOSIT_ID is published; files can no longer change"; exit 1; }

listing () { api "$API/deposit/depositions/$DEPOSIT_ID/files" > "$WORK/files.json"; }
field () {
    python3 -c '
import json, sys
f = next((f for f in json.load(open(sys.argv[1])) if f["filename"] == sys.argv[2]), None)
print("" if f is None else f[sys.argv[3]])
' "$WORK/files.json" "$1" "$2"
}
total () { python3 -c 'import json,sys; print(sum(int(f["filesize"]) for f in json.load(open(sys.argv[1]))))' "$WORK/files.json"; }

shopt -s nullglob
files=("$CLEAN"/*.tar)
[ "${#files[@]}" -gt 0 ] || { echo "nothing in $CLEAN"; exit 1; }
FAILED=""
for src in "${files[@]}"; do
    name="$(basename "$src")"
    md5="$(md5sum "$src" | cut -d' ' -f1)"
    [ ! -s "$src.md5" ] || [ "$(cut -d' ' -f1 "$src.md5")" = "$md5" ] \
        || { echo "$name: md5 $md5 differs from its .md5 file"; FAILED="$FAILED $name"; continue; }
    listing
    have="$(field "$name" checksum)"
    if [ "$have" = "$md5" ]; then echo "ok $name (already replaced)"; continue; fi
    old_id="$(field "$name" id)"
    if [ -n "$old_id" ]; then
        api -X DELETE "$API/deposit/depositions/$DEPOSIT_ID/files/$old_id" > /dev/null
        echo "deleted the draft's $name (md5 $have)"
        listing
    fi
    size="$(stat -c%s "$src")"
    if [ $(( $(total) + size )) -gt "$LIMIT" ]; then
        echo "uploading $name would exceed $LIMIT bytes"; FAILED="$FAILED $name"; continue
    fi
    echo "uploading clean $name ($(numfmt --to=iec "$size"))"
    ok=0
    for attempt in 1 2 3; do
        if api --retry 3 --retry-delay 30 -X PUT --upload-file "$src" \
                -w '\n  HTTP %{http_code} in %{time_total}s (%{speed_upload} B/s)\n' \
                -o "$WORK/put.json" "$BUCKET/$name" > "$WORK/put.log"; then
            cat "$WORK/put.log"
            got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum"].split(":")[-1])' "$WORK/put.json")"
            if [ "$got" = "$md5" ]; then ok=1; break; fi
            echo "  stored md5 $got, expected $md5"
            listing; bad_id="$(field "$name" id)"
            [ -z "$bad_id" ] || api -X DELETE "$API/deposit/depositions/$DEPOSIT_ID/files/$bad_id" > /dev/null
        fi
        echo "  attempt $attempt failed; retrying in 60 s"; sleep 60
    done
    [ "$ok" -eq 1 ] || { echo "  GAVE UP on $name -- the draft now lacks it; re-run"; FAILED="$FAILED $name"; }
done

listing
python3 - "$WORK/files.json" <<'PY'
import json, sys
files = sorted(json.load(open(sys.argv[1])), key=lambda f: f["filename"])
print()
for f in files:
    print(f"  {f['filename']:38s} {int(f['filesize']):>15,} bytes  md5 {f['checksum']}")
print(f"  {len(files)} files, {sum(int(f['filesize']) for f in files) / 1e9:.1f} GB")
PY
[ -z "$FAILED" ] || { echo "FAILED:$FAILED -- re-run this job"; exit 1; }
[ ! -e "$STAGE/.v2_expected_md5.tsv" ] || mv "$STAGE/.v2_expected_md5.tsv" "$STAGE/.v2_expected_md5.tsv.before_scrub"
echo "NOT published. Next: scripts/lrz/zenodo_v2_finish.sh (docs and metadata)."
