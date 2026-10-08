#!/usr/bin/env bash
# Copy the local-only parts of plm_choice to LRZ, verify by md5, and only then report them
# as safe to delete. This script NEVER deletes anything, locally or remotely.
#
#     scripts/lrz/offload_local_only.sh           # dry run: show what would move
#     scripts/lrz/offload_local_only.sh --go      # copy (resumable; safe to re-run)
#     scripts/lrz/offload_local_only.sh --verify  # md5 both sides, print a per-path verdict
#
# Why this exists. The 2026-10-03 audit found ~45 GB of local data already md5-identical on
# LRZ and ~55 GB that exists ONLY on this Mac -- no cluster copy, no Zenodo copy, and for the
# uniref50 .db files not even the .json they were built from, which was deleted on 2026-07-29.
# That is not a keep-or-delete choice, it is a move: push it, prove it arrived, then delete.
set -uo pipefail

REMOTE="${LRZ_HOST:-ai}"
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
ARCHIVE="$DSS/plm_choice_archive_2026-10-08"
UNKNOWN="$DSS/unknown_unknown"
SRC="${SRC_ROOT:-$HOME/Documents/projects/plm_choice}"

# Each entry is "<destination root>|<path relative to the repo root>". $UNKNOWN is kept as a
# destination because the "unknown unknown" manuscript may yet need a tree of its own, but
# nothing is filed there now: the UniRef snapshots it was created for are being deleted
# rather than archived.
# Whole directories, because a half-copied tree is worse than an untouched one.
ENTRIES=(
    # What is left after the 2026-10-08 investigation. Everything else that was on this list
    # turned out not to need archiving, and the reason is recorded next to each removal:
    #
    #   data/raw/2024_new (40.6 GB)  DROPPED -- UniProt still serves release-2024_01 and
    #       2025_01 (checked, HTTP 200), and Tobias will re-download the latest anyway.
    #   data/backup (5.8 GB)         DROPPED -- 23 CSVs, pre-parquet era, zero code references.
    #   foldcomp/afdb_swissprot_v4   DROPPED by the --exclude below -- public AlphaFold DB file.
    #   wandb/, tmp_*/               DROPPED by --exclude -- telemetry and scratch.
    #   sets/{train,val,test}.parquet  ALREADY ON LRZ, md5-proven, under the name
    #       sprot_pre2024_e1/sets/. The earlier audit compared exact paths and missed this.
    #
    # models/ stays because its checkpoints are NOT reproducible: training sets no
    # deterministic=True, and a sampled checkpoint differs from the nearest LRZ grid
    # (epoch=47/step=224064 against epoch=46/step=121401, different md5). Minus wandb it is
    # ~3 GB, not 15.
    "$ARCHIVE|models"
    "$ARCHIVE|data/interm/sprot_pre2024"
)

mode="${1:---dry}"

case "$mode" in
  --dry)
    echo "DRY RUN -- nothing is copied."
    for e in "${ENTRIES[@]}"; do
        d="${e%%|*}"; p="${e#*|}"
        [ -e "$SRC/$p" ] || { echo "  MISSING LOCALLY  $p"; continue; }
        printf '  %-30s %6s  ->  %s\n' "$p" "$(du -sh "$SRC/$p" 2>/dev/null | cut -f1)" "$(basename "$d")"
    done
    echo
    echo "Re-run with --go to copy, then --verify before deleting anything."
    ;;

  --go)
    ssh "$REMOTE" "mkdir -p '$ARCHIVE' '$UNKNOWN'" || exit 1
    # The README travels with the data: in November nobody will remember why this tree exists.
    ssh "$REMOTE" "cat > '$ARCHIVE/README.md'" <<'NOTE'
# plm_choice — local-only archive, offloaded 2026-10-08

Everything here existed **only** on Tobias's Mac. It was copied here (not moved) so the Mac
could be freed; each path was md5-verified on both sides before any local deletion.

| path | why it was local-only |
|---|---|
| `models/` | 15 GB. The ORIGINAL submission probes (`sprot_pre2024_subset`, `_subset_pca`, `_subset_run1`, `sprot_pre2024`, `sprot_train`, `train_sub`). LRZ's own model dirs are a *disjoint* set. Training sets no `deterministic=True`, so a retrain is equivalent-but-not-identical. Superseded by `full100_p10` / `pca_cohort` as of 2026-10-02, but kept because deleting them makes the submitted numbers unreproducible. |
| `data/backup/` | 23 CSVs dated 2024-12-27 to 2025-07-17, pre-parquet era, **zero code references** anywhere in `src/`, `scripts/` or `tests/`. |
| `data/interm/sprot_pre2024/` | 533 local-only files (foldcomp, foldseek, mmseqs, data_split_*). 4 further files were already on LRZ and are copied here too so this tree stands alone. |

**Not here:** `data/raw/2024_new/` (the UniRef snapshots, 40.6 GB) left plm_choice and lives in
`../unknown_unknown/`. It backs the November manuscript, not this paper.
NOTE
    ssh "$REMOTE" "cat > '$UNKNOWN/README.md'" <<'NOTE2'
# unknown unknown — UniRef snapshots, offloaded from plm_choice 2026-10-08

`data/raw/2024_new/uniref50_2024_01.db` (28.68 GB) and `uniref50_2025_01.db` (11.95 GB).

These left the plm_choice project. They are filed here because they belong to the
**"unknown unknown" manuscript**, which Tobias resumes in **November 2026** — not to the pLM
benchmark paper, even though plm_choice is the tree they were copied out of.

**They are the last copy.** The `.json` files they were built from were deleted from the Mac on
2026-07-29 as part of a 135.9 GB reclaim; only the `.db` is read by any code. Re-downloading the
2024_01 snapshot from UniProt may not be possible — old releases are not guaranteed to stay served.

Their only previous role in plm_choice was the New2024 post-cutoff check (5,466 pairs over 673
proteins), which is already computed and reported, so nothing in that paper is pending on them.

Copied, md5-verified, and only then deleted from the Mac. Nothing was moved without a checksum.
NOTE2
    for e in "${ENTRIES[@]}"; do
        d="${e%%|*}"; p="${e#*|}"
        [ -e "$SRC/$p" ] || { echo "SKIP (missing) $p"; continue; }
        echo "=== $p  ->  $(basename "$d") ==="
        ssh "$REMOTE" "mkdir -p '$d/$(dirname "$p")'"
        # -a preserves times so a re-run skips what already landed; --partial keeps a half
        # file so an interrupted 28 GB transfer resumes instead of restarting.
        # Exclusions, measured rather than assumed:
        #   wandb/     81.3% of local models/ (13.02 GB of 16.01) is offline training telemetry,
        #              against 0.30 GB of actual checkpoints. Same finding as the Zenodo tars.
        #   tmp_*/     2.36 GB of MMseqs/Foldseek scratch databases under data_split_*. The
        #              2026-07-29 reclaim already deleted 38.6 GB of exactly this class.
        rsync -a --partial --human-readable --info=progress2 \
              --exclude='wandb/' --exclude='*.wandb' --exclude='last.ckpt' \
              --exclude='tmp_clustering/' --exclude='tmp_*/' \
              --exclude='foldcomp/afdb_swissprot_v4' \
              "$SRC/$p/" "$REMOTE:$d/$p/" || echo "  rsync returned $? for $p"
    done
    echo
    echo "Copied. Now run --verify. Do NOT delete anything until it reports OK."
    ;;

  --verify)
    # LC_ALL=C on BOTH sorts: macOS and Linux collate mixed-case names differently (SUMMARY.md
    # sorts before cache.txt on one, after it on the other), which misaligns a line diff and
    # reports a mismatch between identical trees.
    echo "md5 both sides, per file. This reads every byte on both ends."
    rc=0
    for e in "${ENTRIES[@]}"; do
        d="${e%%|*}"; p="${e#*|}"
        # A path that is not there cannot be verified. Skipping it used to print "All paths
        # verified" over a check that never ran.
        if [ ! -e "$SRC/$p" ]; then echo "=== $p: MISSING LOCALLY -- nothing to verify"; rc=1; continue; fi
        echo "=== $p  ($(basename "$d")) ==="
        ( cd "$SRC/$p" && find . -type f \
            ! -path '*/wandb/*' ! -name '*.wandb' ! -name 'last.ckpt' \
            ! -path '*/tmp_clustering/*' ! -path '*/tmp_*/*' \
            ! -name 'afdb_swissprot_v4' \
            -exec md5 -q {} \; -print \
            | paste - - | awk '{print $1"  "$2}' | LC_ALL=C sort -k2 ) > "/tmp/.loc_$$"
        ssh "$REMOTE" "cd '$d/$p' || exit 9; find . -type f -exec md5sum {} \; | LC_ALL=C sort -k2" \
            | awk '{print $1"  "$2}' | LC_ALL=C sort -k2 > "/tmp/.rem_$$"
        n=$(grep -c . "/tmp/.loc_$$")
        # Zero files on both sides compares equal. That is not a verification: on 2026-10-08 a
        # check of exactly this shape reported "VERIFY OK: 0 files identical" one step before
        # a tree it had never copied would have been deleted.
        if [ "$n" -eq 0 ]; then
            echo "  ** NOTHING CHECKED (0 local files) -- refusing"; rc=1
        elif diff -q "/tmp/.loc_$$" "/tmp/.rem_$$" >/dev/null; then
            echo "  OK  $(wc -l < "/tmp/.loc_$$" | tr -d ' ') files identical"
        else
            echo "  ** MISMATCH -- do not delete $p **"
            diff "/tmp/.loc_$$" "/tmp/.rem_$$" | head -10
            rc=1
        fi
        rm -f "/tmp/.loc_$$" "/tmp/.rem_$$"
    done
    [ "$rc" -eq 0 ] && echo && echo "All paths verified. These are now safe to delete locally."
    exit "$rc"
    ;;

  *) echo "usage: $0 [--dry|--go|--verify]"; exit 2 ;;
esac
