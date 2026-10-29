#!/usr/bin/env bash
# Finish version 2 of the Zenodo record: check the files carry their published names
# (zenodo_v2_rename.sbatch), upload the documentation from docs/zenodo/, remove the 2025
# supplementary_data.zip the new version inherited, and write the metadata from
# docs/zenodo/metadata.json. Small API calls only, so it runs on the login node:
#
#     DRY_RUN=1 scripts/lrz/zenodo_v2_finish.sh     # show what would change
#     scripts/lrz/zenodo_v2_finish.sh               # change the draft
#
# WHAT THIS DOES NOT DO: publish. Publishing is irreversible (files freeze, the version DOI
# goes live) and belongs to a human with the draft open in a browser.
#
# The token is read from ~/.zenodo_token and handed to curl through a header file, never on
# the command line, where every user on the node could read it with ps.

set -euo pipefail

DEPOSIT_ID="${DEPOSIT_ID:-22935411}"
API="https://zenodo.org/api"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DOCS="$REPO/docs/zenodo"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.zenodo_token}"
DRY_RUN="${DRY_RUN:-}"

[ -r "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE"; exit 2; }
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
printf 'Authorization: Bearer %s\n' "$(tr -d ' \n' < "$TOKEN_FILE")" > "$WORK/auth"
api () { curl -sS --fail-with-body -H @"$WORK/auth" "$@"; }

api "$API/deposit/depositions/$DEPOSIT_ID" > "$WORK/dep.json"
STATE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$WORK/dep.json")"
[ "$STATE" != "done" ] || { echo "deposit $DEPOSIT_ID is already published; nothing to do"; exit 1; }
BUCKET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["links"]["bucket"])' "$WORK/dep.json")"

# 0. Published names. Files are built and uploaded under their working names (the ones
#    zenodo_upload.sbatch and zenodo_stage_v2.sh stage) and given their published names by
#    zenodo_v2_rename.sbatch (docs/zenodo/file_names.tsv). The docs describe the published
#    names, so stop here while any working name is left.
api "$API/deposit/depositions/$DEPOSIT_ID/files" > "$WORK/files.json"
LEFT="$(python3 -c '
import json, sys
have = {f["filename"] for f in json.load(open(sys.argv[1]))}
old = [l.split("\t")[0] for l in open(sys.argv[2]) if l.strip() and not l.startswith("#")]
print(" ".join(o for o in old if o in have))
' "$WORK/files.json" "$DOCS/file_names.tsv")"
if [ -n "$LEFT" ]; then
    echo "still under working names:$LEFT"
    echo "run first: sbatch scripts/lrz/zenodo_v2_rename.sbatch"
    [ -n "$DRY_RUN" ] || exit 1
fi

# 1. Documentation: PUT to the bucket under the same name replaces the inherited 2025 file.
for name in README.md DATA_DICTIONARY.md CITATION.cff; do
    want="$(md5sum "$DOCS/$name" | cut -d' ' -f1)"
    if [ -n "$DRY_RUN" ]; then echo "would upload $name (md5 $want)"; continue; fi
    api -X PUT --upload-file "$DOCS/$name" "$BUCKET/$name" > "$WORK/put.json"
    got="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checksum"].split(":")[-1])' "$WORK/put.json")"
    [ "$got" = "$want" ] || { echo "upload of $name stored md5 $got, expected $want"; exit 1; }
    echo "uploaded $name (md5 $got)"
done

# 2. The 2025 archive: its pair sets, probes and figures are superseded by this version's
#    files. Version 1 (10.5281/zenodo.17469268) keeps it.
api "$API/deposit/depositions/$DEPOSIT_ID/files" > "$WORK/files.json"
ZIP_ID="$(python3 -c '
import json, sys
print(next((f["id"] for f in json.load(open(sys.argv[1])) if f["filename"] == "supplementary_data.zip"), ""))
' "$WORK/files.json")"
if [ -z "$ZIP_ID" ]; then
    echo "supplementary_data.zip: not in the draft"
elif [ -n "$DRY_RUN" ]; then
    echo "would delete supplementary_data.zip ($ZIP_ID)"
else
    api -X DELETE "$API/deposit/depositions/$DEPOSIT_ID/files/$ZIP_ID" > /dev/null
    echo "deleted supplementary_data.zip"
fi

# 3. Metadata: a PUT replaces the whole record, so start from the current metadata (which
#    carries the reserved DOI) and overlay docs/zenodo/metadata.json.
python3 - "$WORK/dep.json" "$DOCS/metadata.json" "$DOCS/description.html" > "$WORK/meta.json" <<'PY'
import json, sys
current = json.load(open(sys.argv[1]))["metadata"]
new = json.load(open(sys.argv[2]))
new["description"] = open(sys.argv[3]).read().strip()
current.update(new)
json.dump({"metadata": current}, sys.stdout, ensure_ascii=False)
PY
if [ -n "$DRY_RUN" ]; then
    echo "would set metadata:"; python3 -m json.tool "$WORK/meta.json" | head -60
else
    api -X PUT -H "Content-Type: application/json" --data @"$WORK/meta.json" \
        "$API/deposit/depositions/$DEPOSIT_ID" > /dev/null
    echo "metadata written"
fi

# 4. The record page previews README.md and lists the files in the README's order; without
#    this Zenodo previews the first previewable file by name (DATA_DICTIONARY.md). Only the
#    InvenioRDM API sets these, and its PUT replaces the whole draft: so the draft is read in
#    that format, only `files` is changed, and the descriptive metadata must match before and after.
python3 - "$WORK/auth" "$DEPOSIT_ID" "$DOCS/file_names.tsv" "${DRY_RUN:-}" <<'PY'
import json, sys, urllib.error, urllib.request
auth = open(sys.argv[1]).read().split(":", 1)[1].strip()
dep, tsv, dry = sys.argv[2], sys.argv[3], sys.argv[4]
draft = f"https://zenodo.org/api/records/{dep}/draft"
rdm = "application/vnd.inveniordm.v1+json"

def call(method, url, body=None, accept=rdm):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": auth, "Accept": accept, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {url}: HTTP {e.code} {e.read()[:500]!r}")

def described():
    m = call("GET", f"https://zenodo.org/api/deposit/depositions/{dep}", accept="application/json")["metadata"]
    keys = ("title", "creators", "description", "version", "publication_date", "upload_type", "license",
            "access_right", "keywords", "related_identifiers", "prereserve_doi")
    return {k: m.get(k) for k in keys}

order = ["README.md", "DATA_DICTIONARY.md", "CITATION.cff"] + [
    line.split("\t")[1].strip() for line in open(tsv) if line.strip() and not line.startswith("#")]
d = call("GET", draft)
entries = d["files"].get("entries") or {}
have = set(entries) if isinstance(entries, dict) else {e["key"] for e in entries}
if have != set(order):
    sys.exit(f"draft files differ from the expected list: {sorted(have ^ set(order))}")
if d["files"].get("default_preview") == "README.md" and d["files"].get("order") == order:
    print("preview and file order already set"); sys.exit()
if dry:
    print("would set default preview README.md and the file order:", ", ".join(order)); sys.exit()
before = described()
body = {k: d[k] for k in ("access", "metadata", "pids", "custom_fields") if k in d}
body["files"] = {"enabled": True, "default_preview": "README.md", "order": order}
out = call("PUT", draft, body)
after = described()
moved = [k for k in before if before[k] != after[k]]
if moved:
    sys.exit(f"metadata changed by the files update: {moved} -- check the draft before publishing")
if out["files"].get("default_preview") != "README.md":
    sys.exit(f"default preview not set: {out['files'].get('default_preview')!r}")
print("default preview README.md; file order set; metadata unchanged")
PY

# 5. What the draft now holds.
api "$API/deposit/depositions/$DEPOSIT_ID" > "$WORK/dep.json"
python3 - "$WORK/dep.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d["metadata"]
print()
print("title:  ", m.get("title")); print("version:", m.get("version"), "| type:", m.get("upload_type"),
      "| date:", m.get("publication_date"), "| DOI:", m.get("prereserve_doi", {}).get("doi"))
for c in m.get("creators", []):
    print("  ", c.get("name"), "|", c.get("orcid"), "|", c.get("affiliation"))
files = sorted(d.get("files", []), key=lambda f: f["filename"])
for f in files:
    print(f"  {f['filename']:34s} {int(f['filesize']):>15,} bytes  md5 {f['checksum']}")
print(f"  {len(files)} files, {sum(int(f['filesize']) for f in files) / 1e9:.1f} GB")
PY
echo
echo "NOT published. Review at https://zenodo.org/uploads/$DEPOSIT_ID and publish there."
