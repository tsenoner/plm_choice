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
DEST="$DSS/plm_choice_archive_2026-10-08"
SRC="${SRC_ROOT:-$HOME/Documents/projects/plm_choice}"

# Each entry is a path relative to the repo root. Whole directories, because a half-copied
# tree is worse than an untouched one and rsync makes the duplicates cheap.
PATHS=(
    "data/raw/2024_new"              # 40.6 GB -- uniref50_{2024_01,2025_01}.db, last copy
    "models"                         # 15.0 GB -- original submission probes, not reproducible
    "data/backup"                    #  5.8 GB -- pre-parquet CSVs, zero code references
    "data/interm/sprot_pre2024"      #  7.1 GB local-only (+4.1 GB already on LRZ, copied anyway
                                     #            so the archive is self-contained)
)

mode="${1:---dry}"

case "$mode" in
  --dry)
    echo "DRY RUN -- nothing is copied. Destination: $REMOTE:$DEST"
    for p in "${PATHS[@]}"; do
        [ -e "$SRC/$p" ] || { echo "  MISSING LOCALLY  $p"; continue; }
        printf '  %-32s %s\n' "$p" "$(du -sh "$SRC/$p" 2>/dev/null | cut -f1)"
    done
    echo
    echo "Re-run with --go to copy, then --verify before deleting anything."
    ;;

  --go)
    ssh "$REMOTE" "mkdir -p '$DEST'" || exit 1
    # The README travels with the data: in November nobody will remember why this tree exists.
    ssh "$REMOTE" "cat > '$DEST/README.md'" <<'NOTE'
# plm_choice — local-only archive, offloaded 2026-10-08

Everything here existed **only** on Tobias's Mac. It was copied here (not moved) so the Mac
could be freed; each path was md5-verified on both sides before any local deletion.

| path | why it was local-only |
|---|---|
| `data/raw/2024_new/` | `uniref50_{2024_01,2025_01}.db`, 40.6 GB. The `.json` files these were built from were deleted on 2026-07-29, so these are the last copy. Code reads the `.db`, never the `.json`. Backs the New2024 post-cutoff check in the paper. |
| `models/` | 15 GB. The ORIGINAL submission probes (`sprot_pre2024_subset`, `_subset_pca`, `_subset_run1`, `sprot_pre2024`, `sprot_train`, `train_sub`). LRZ's own model dirs are a *disjoint* set. Training sets no `deterministic=True`, so a retrain is equivalent-but-not-identical. Superseded by `full100_p10` / `pca_cohort` as of 2026-10-02, but kept because deleting them makes the submitted numbers unreproducible. |
| `data/backup/` | 23 CSVs dated 2024-12-27 to 2025-07-17, pre-parquet era, **zero code references** anywhere in `src/`, `scripts/` or `tests/`. |
| `data/interm/sprot_pre2024/` | 533 local-only files (foldcomp, foldseek, mmseqs, data_split_*). 4 further files were already on LRZ and are copied here too so this tree stands alone. |

**Note on provenance:** some of this may belong with the "into the unknown" manuscript rather
than plm_choice. It is filed under plm_choice because that is the tree it was taken from and
that is what the file paths say. Re-filing is one `mv` on this filesystem.
NOTE
    for p in "${PATHS[@]}"; do
        [ -e "$SRC/$p" ] || { echo "SKIP (missing) $p"; continue; }
        echo "=== $p ==="
        ssh "$REMOTE" "mkdir -p '$DEST/$(dirname "$p")'"
        # -a preserves times so a re-run skips what already landed; --partial keeps a half
        # file so an interrupted 28 GB transfer resumes instead of restarting.
        rsync -a --partial --human-readable --info=progress2 \
              "$SRC/$p/" "$REMOTE:$DEST/$p/" || echo "  rsync returned $? for $p"
    done
    echo
    echo "Copied. Now run --verify. Do NOT delete anything until it reports OK."
    ;;

  --verify)
    echo "md5 both sides, per file. This reads every byte on both ends."
    rc=0
    for p in "${PATHS[@]}"; do
        [ -e "$SRC/$p" ] || continue
        echo "=== $p ==="
        ( cd "$SRC/$p" && find . -type f -exec md5 -q {} \; -print \
            | paste - - | awk '{print $1"  "$2}' | sort -k2 ) > "/tmp/.loc_$$"
        ssh "$REMOTE" "cd '$DEST/$p' && find . -type f -exec md5sum {} \; | sort -k2" \
            | awk '{print $1"  "$2}' | sort -k2 > "/tmp/.rem_$$"
        if diff -q "/tmp/.loc_$$" "/tmp/.rem_$$" >/dev/null; then
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
