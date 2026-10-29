#!/bin/bash
# Give the files of the Zenodo v2 draft their published names (docs/zenodo/file_names.tsv).
#
# Zenodo cannot rename a draft file: the documented PUT /deposit/depositions/:id/files/:id
# returns 405 since the move to InvenioRDM, and the draft-files API rejects a new "key"
# (both tried 2026-10-29). So each file is uploaded again under its new name and the old copy
# is deleted, one file at a time, because a record may not exceed 50 GB and this one holds 43.
#
#     sbatch scripts/lrz/zenodo_v2_rename.sbatch          # from the repo root on the cluster
#
# Safe to re-run. The md5 of every file is read from the draft on the first run and kept in
# $STAGE/.v2_expected_md5.tsv, so a resumed run still knows it after an old copy is gone. A file
# already under its new name with that md5 is skipped; the old copy is deleted only after
# Zenodo reports the new copy's md5 equal to it. Nothing here publishes.
#
#SBATCH -J zenodo-rename
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

DEPOSIT_ID="${DEPOSIT_ID:-22935411}"
API="https://zenodo.org/api"
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
STAGE="$DSS/zenodo_stage"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
MAP="$REPO/docs/zenodo/file_names.tsv"
EXPECT="$STAGE/.v2_expected_md5.tsv"
LIMIT=50000000000
TOKEN_FILE="${TOKEN_FILE:-$HOME/.zenodo_token}"

[ -r "$MAP" ] || { echo "no map at $MAP -- submit from the repo root"; exit 2; }
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
# field <name> <id|checksum|filesize> : that field of the draft file called <name>, or empty
field () {
    python3 -c '
import json, sys
f = next((f for f in json.load(open(sys.argv[1])) if f["filename"] == sys.argv[2]), None)
print("" if f is None else (int(f[sys.argv[3]]) if sys.argv[3] == "filesize" else f[sys.argv[3]]))
' "$WORK/files.json" "$1" "$2"
}
total () { python3 -c 'import json,sys; print(sum(int(f["filesize"]) for f in json.load(open(sys.argv[1]))))' "$WORK/files.json"; }

# 1. What every file must hold, read once from the draft while the old names still exist.
listing
if [ ! -s "$EXPECT" ]; then
    grep -v '^#' "$MAP" | while IFS=$'\t' read -r old new; do
        md5="$(field "$old" checksum)"; size="$(field "$old" filesize)"
        [ -n "$md5" ] || { echo "$old is not in the draft; cannot record its md5"; exit 1; }
        printf '%s\t%s\t%s\t%s\n' "$old" "$new" "$md5" "$size"
    done > "$EXPECT.part"
    mv "$EXPECT.part" "$EXPECT"
    echo "recorded the md5 of $(wc -l < "$EXPECT") files in $EXPECT"
fi

# 2. One file at a time: upload under the new name, compare md5, delete the old copy.
FAILED=""
while IFS=$'\t' read -r -u 3 old new md5 size; do
    listing
    new_md5="$(field "$new" checksum)"; new_id="$(field "$new" id)"
    if [ -n "$new_md5" ] && [ "$new_md5" != "$md5" ]; then
        echo "$new holds md5 $new_md5, expected $md5: removing it to upload again"
        api -X DELETE "$API/deposit/depositions/$DEPOSIT_ID/files/$new_id" > /dev/null
        new_md5=""; listing
    fi
    if [ -z "$new_md5" ]; then
        src="$STAGE/$old"
        [ -e "$src" ] || { echo "no source for $old at $src"; FAILED="$FAILED $new"; continue; }
        have="$(stat -Lc%s "$src")"
        [ "$have" = "$size" ] || { echo "$src is $have bytes, the draft's $old $size"; FAILED="$FAILED $new"; continue; }
        now="$(total)"
        if [ $((now + size)) -gt "$LIMIT" ]; then
            echo "uploading $new would take the record to $((now + size)) bytes, over $LIMIT"; FAILED="$FAILED $new"; continue
        fi
        echo "uploading $old as $new ($(numfmt --to=iec "$size"))"
        ok=0
        for attempt in 1 2 3; do
            if api --retry 3 --retry-delay 30 -X PUT --upload-file "$src" \
                    -w '\n  HTTP %{http_code} in %{time_total}s (%{speed_upload} B/s)\n' \
                    -o "$WORK/put.json" "$BUCKET/$new" > "$WORK/put.log"; then
                cat "$WORK/put.log"
                got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum"].split(":")[-1])' "$WORK/put.json")"
                if [ "$got" = "$md5" ]; then ok=1; break; fi
                echo "  stored md5 $got, expected $md5"
            fi
            echo "  attempt $attempt failed; retrying in 60 s"; sleep 60
        done
        [ "$ok" -eq 1 ] || { echo "  GAVE UP on $new; its old copy stays"; FAILED="$FAILED $new"; continue; }
        listing
    else
        echo "ok $new (already uploaded)"
    fi
    old_id="$(field "$old" id)"
    if [ -n "$old_id" ]; then
        api -X DELETE "$API/deposit/depositions/$DEPOSIT_ID/files/$old_id" > /dev/null
        echo "  deleted $old"
    fi
done 3< "$EXPECT"

# 3. The draft must now hold exactly the published names, each with its recorded md5.
listing
python3 - "$WORK/files.json" "$EXPECT" <<'PY'
import json, sys
files = {f["filename"]: f for f in json.load(open(sys.argv[1]))}
expect = [line.rstrip("\n").split("\t") for line in open(sys.argv[2])]
bad = [new for old, new, md5, size in expect if files.get(new, {}).get("checksum") != md5]
left = [old for old, new, md5, size in expect if old in files]
print()
for name in sorted(files):
    print(f"  {name:38s} {int(files[name]['filesize']):>15,} bytes  md5 {files[name]['checksum']}")
print(f"  {len(files)} files, {sum(int(f['filesize']) for f in files.values()) / 1e9:.1f} GB")
if bad or left:
    print("\nNOT DONE -- missing or wrong:", bad, "| old names left:", left)
    sys.exit(1)
print("\nall", len(expect), "files under their published names, md5 as before")
PY
[ -z "$FAILED" ] || { echo "FAILED:$FAILED -- re-run this job"; exit 1; }
echo "NOT published. Next: scripts/lrz/zenodo_v2_finish.sh (docs and metadata)."
