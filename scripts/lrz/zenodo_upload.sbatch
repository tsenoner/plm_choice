#!/bin/bash
# Upload the deposit to Zenodo straight from the cluster, so ~33 GB never travels via a
# laptop. Runs as a job because the login nodes are capped at a few minutes and this is
# measured in hours.
#
#     # once, by hand, on the cluster -- NEVER in git, NEVER in a commit message:
#     printf '%s' 'YOUR_TOKEN' > ~/.zenodo_token && chmod 600 ~/.zenodo_token
#
#     # then, to create a new version of the existing record and upload into its draft:
#     sbatch scripts/lrz/zenodo_upload.sbatch
#
#     # to add files to a draft that already exists (e.g. the probe weights, later):
#     DEPOSIT_ID=<id printed by the first run> sbatch scripts/lrz/zenodo_upload.sbatch
#
# WHAT THIS DOES NOT DO: publish. It leaves the deposit as a DRAFT, which is reversible
# and which already reserves the DOI the manuscript cites. Publishing is a deliberate,
# irreversible act and belongs to a human with the record open in a browser.
#
#SBATCH -J zenodo-upload
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
PROC="$DSS/plm_choice_data/data/processed"
CONCEPT_RECORD="${CONCEPT_RECORD:-17469268}"        # the existing Zenodo record
API="https://zenodo.org/api"
STAGE="$DSS/zenodo_stage"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.zenodo_token}"

[ -r "$TOKEN_FILE" ] || { echo "no token at $TOKEN_FILE -- see the header"; exit 2; }
TOKEN="$(tr -d ' \n' < "$TOKEN_FILE")"
# The token must never reach the log: curl gets it from a header file, not from argv,
# because argv is visible to every user on the node via ps.
AUTH_HEADER="$(mktemp)"; trap 'rm -f "$AUTH_HEADER"' EXIT
printf 'Authorization: Bearer %s\n' "$TOKEN" > "$AUTH_HEADER"
api () { curl -sS -H @"$AUTH_HEADER" "$@"; }

# ---------------------------------------------------------------------------
# 1. The draft: either a new version of the record, or one we were given.
# ---------------------------------------------------------------------------
if [ -n "${DEPOSIT_ID:-}" ]; then
    echo "using existing draft $DEPOSIT_ID"
else
    echo "creating a new version of record $CONCEPT_RECORD"
    NEWV="$(api -X POST "$API/deposit/depositions/$CONCEPT_RECORD/actions/newversion")"
    DEPOSIT_ID="$(printf '%s' "$NEWV" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["links"]["latest_draft"].rstrip("/").rsplit("/",1)[-1])')"
    echo "new draft: $DEPOSIT_ID"
fi

DEP="$(api "$API/deposit/depositions/$DEPOSIT_ID")"
BUCKET="$(printf '%s' "$DEP" | python3 -c 'import sys,json; print(json.load(sys.stdin)["links"]["bucket"])')"
DOI="$(printf '%s' "$DEP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["metadata"].get("prereserve_doi",{}).get("doi") or d.get("doi") or "not reserved")')"
echo "bucket: $BUCKET"
echo "reserved version DOI: $DOI"
echo "the manuscript cites the CONCEPT DOI (10.5281/zenodo.17469267), which resolves here once published"

# ---------------------------------------------------------------------------
# 2. What goes in. Archives are built once into $STAGE and reused, so a re-run
#    after a network failure does not rebuild 31 GB of tar.
# ---------------------------------------------------------------------------
mkdir -p "$STAGE"
build_tar () {   # build_tar <name> <parent-dir> <child>
    local name="$1" parent="$2" child="$3"
    if [ -s "$STAGE/$name" ]; then echo "  reuse $name"; return; fi
    echo "  building $name"
    tar -C "$parent" -cf "$STAGE/$name.part" "$child"
    mv "$STAGE/$name.part" "$STAGE/$name"
}

# A new version starts as a COPY of the previous version's files: record 17469268 already
# holds 15 embedding .h5 files (~29 GB) plus supplementary_data.zip, and they are inherited
# unless we overwrite them.
#
# WITH_EMBEDDINGS=1 replaces the .h5 files with the cohort this paper used. They are not
# the same files -- the deposited esm2_3b.h5 is 4,616 MB against 5,745 MB on the cluster,
# a different protein cut -- so publishing without replacing them would ship embeddings
# that are not the ones behind the numbers. They go up under their EXISTING names, so each
# one overwrites its predecessor in place: the record keeps its 15-file shape instead of
# growing a parallel tar, and no reader ends up with two versions of esm1b.h5.
build_tar pair_tables_splits.tar      "$PROC"                        sprot_pre2024_e1
build_tar identical_sequence_mask.tar "$DSS"                         ridge_identical
[ -d "$DSS/pair_table_counts" ] && build_tar pair_table_counts.tar   "$DSS" pair_table_counts

# The probe weights and their metrics: small, and a reason to deposit at all -- a reader
# can score a new pair set without a GPU.
#
# "The directory exists" is NOT the test for whether they are ready. A grid in progress
# has a models/ directory from its first cell, and tarring it gave
# "tar: ... file changed as we read it" while Slurm was writing into it -- which would
# have deposited a partial grid that looks complete. The test is the metrics file count:
# 45 cells x 2 read-outs = 90 per dataset. Set EXPECT_METRICS=0 to override.
EXPECT_METRICS="${EXPECT_METRICS:-90}"
for d in full100_p10 pca100_p10; do
    dir="$HOME/plm_choice/models/$d"
    if [ ! -d "$dir" ]; then
        echo "  SKIP probes_$d.tar -- $dir does not exist yet"
        continue
    fi
    have="$(find "$dir" -name '*_metrics.txt' | wc -l)"
    if [ "$have" -lt "$EXPECT_METRICS" ]; then
        echo "  SKIP probes_$d.tar -- $have/$EXPECT_METRICS metrics files; the grid is still running"
        continue
    fi
    if squeue -u "$USER" -h -o '%j' | grep -q probe-grid; then
        echo "  SKIP probes_$d.tar -- probe-grid jobs are still queued or running"
        continue
    fi
    build_tar "probes_$d.tar" "$HOME/plm_choice/models" "$d"
done

# ---------------------------------------------------------------------------
# 3. Upload. One PUT per file; Zenodo has no resume, so a failed file is retried
#    whole, and an already-uploaded file of the same size is skipped.
# ---------------------------------------------------------------------------
if [ -n "${WITH_EMBEDDINGS:-}" ]; then
    echo "staging the cohort embeddings for in-place replacement"
    for h5 in "$PROC/sprot_pre2024/embeddings_cohort2k"/*.h5; do
        ln -sfn "$h5" "$STAGE/$(basename "$h5")"
    done
fi

UPLOADED="$(api "$API/deposit/depositions/$DEPOSIT_ID/files" | python3 -c 'import sys,json; print(" ".join(f["filename"] for f in json.load(sys.stdin)))' || echo "")"
echo "already in the draft: ${UPLOADED:-nothing}"

shopt -s nullglob
for f in "$STAGE"/*.tar "$STAGE"/*.h5; do
    name="$(basename "$f")"
    # An inherited file of the same name is the one we are replacing, so "already there"
    # is not a reason to skip: only skip what THIS deposit already received, tracked in
    # a marker file, so an interrupted run resumes without re-sending 5 GB.
    if [ -f "$STAGE/.done_$name" ]; then echo "skip $name (uploaded by an earlier run)"; continue; fi
    size="$(stat -c%s "$f")"
    echo "uploading $name ($(numfmt --to=iec "$size"))"
    for attempt in 1 2 3; do
        if curl -sS -H @"$AUTH_HEADER" --retry 3 --retry-delay 30 \
                -o /dev/null -w '  HTTP %{http_code} in %{time_total}s (%{speed_upload} B/s)\n' \
                -X PUT --upload-file "$f" "$BUCKET/$name"; then
            touch "$STAGE/.done_$name"
            break
        fi
        echo "  attempt $attempt failed; retrying"
        sleep 60
    done
done
shopt -u nullglob

echo
echo "draft $DEPOSIT_ID now holds:"
api "$API/deposit/depositions/$DEPOSIT_ID/files" | python3 -c 'import sys,json;[print(f"  {x[\"filename\"]:34} {int(x[\"filesize\"]):>14,} bytes") for x in json.load(sys.stdin)]'
echo
echo "NOT published. Review at https://zenodo.org/deposit/$DEPOSIT_ID and publish there."
