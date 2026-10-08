#!/usr/bin/env bash
# Delete local files that are provably safe to delete. Dry run by default.
#
#     scripts/lrz/reclaim_local.sh            # show what would go, verify nothing
#     scripts/lrz/reclaim_local.sh --check    # md5 every candidate against LRZ, delete nothing
#     scripts/lrz/reclaim_local.sh --go       # verify, then delete ONLY what verified
#
# The rule, learned on 2026-07-29: re-verify immediately before removal. An audit from an hour
# ago is not evidence about the file in front of you. --go re-checks every byte itself; it does
# not trust --check, this script's own earlier run, or docs/NEXT.md.
set -uo pipefail

REMOTE="${LRZ_HOST:-ai}"
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
ARCHIVE="$DSS/plm_choice_archive_2026-10-08"
SRC="${SRC_ROOT:-$HOME/Documents/projects/plm_choice}"
mode="${1:---dry}"

# --- Group A: must match a file on LRZ, byte for byte, before it is deleted -------------
# "<local path>|<remote path>" -- the remote path is explicit because LRZ stores several of
# these under different directory names (sets/ lives under sprot_pre2024_e1/, not
# sprot_pre2024/), and the 2026-10-03 audit called them local-only for exactly that reason.
VERIFIED=(
  "data/processed/sprot_pre2024/sets/train.parquet|$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets/train.parquet"
  "data/processed/sprot_pre2024/sets/val.parquet|$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets/val.parquet"
  "data/processed/sprot_pre2024/sets/test.parquet|$DSS/plm_choice_data/data/processed/sprot_pre2024_e1/sets/test.parquet"
  "data/processed/sprot_pre2024/sets/train_ext.parquet|$DSS/plm_choice_data/data/processed/sprot_pre2024/sets/train_ext.parquet"
  "data/processed/sprot_pre2024/embeddings/prostt5.h5|$DSS/plm_choice_data/data/processed/sprot_pre2024/embeddings/prostt5.h5"
  "data/processed/sprot_pre2024_subset/sets/train_ext.parquet|$DSS/plm_choice_data/data/processed/sprot_pre2024_subset/sets/train_ext.parquet"
  "data/interm/sprot_pre2024/foldseek/afdb_swissprot_v4_all_vs_all.parquet|$ARCHIVE/data/interm/sprot_pre2024/foldseek/afdb_swissprot_v4_all_vs_all.parquet"
  "data/interm/sprot_pre2024/mmseqs/sprot_all_vs_all.parquet|$ARCHIVE/data/interm/sprot_pre2024/mmseqs/sprot_all_vs_all.parquet"
  "data/interm/sprot_pre2024/foldcomp/plddt.tsv|$ARCHIVE/data/interm/sprot_pre2024/foldcomp/plddt.tsv"
)

# --- Group B: deleted on their own merits, with no remote counterpart to check ----------
# Each line is "<path>|<why>". Nothing here is a result.
UNVERIFIED=(
  "data/raw/2024_new|UniRef snapshots. Re-derivable: UniProt still serves release-2024_01 and 2025_01 (checked 2026-10-08, HTTP 200), and the latest release is wanted anyway. Their only product, 2024_novelSeqs_cohort1225.fasta (443 KB), is already computed and kept."
  "data/backup|23 CSVs dated 2024-12 to 2025-07, pre-parquet era, ZERO code references in src/, scripts/ or tests/."
  "data/interm/sprot_pre2024/data_split_foldseek/tmp_clustering|MMseqs/Foldseek scratch. 38.6 GB of the same class was deleted on 2026-07-29."
  "data/interm/sprot_pre2024/data_split_mmseqs/tmp_clustering|As above."
  "data/interm/sprot_pre2024/foldcomp/afdb_swissprot_v4|Public AlphaFold DB FoldComp database; re-downloadable."
)

# --- never touched, whatever else happens ----------------------------------------------
# models/ is absent on purpose: its checkpoints are NOT reproducible (no deterministic=True)
# and they are deleted only by hand, after offload_local_only.sh --verify passes.
PROTECTED=(manuscript docs src scripts tests .venv .git out freeze bin notebooks)

say() { printf '%s\n' "$*"; }
human() { du -sh "$1" 2>/dev/null | cut -f1; }

for p in "${PROTECTED[@]}"; do
    for v in "${VERIFIED[@]}" "${UNVERIFIED[@]}"; do
        case "${v%%|*}" in "$p"|"$p"/*)
            say "REFUSING: $p is protected but appears in the delete list"; exit 3 ;;
        esac
    done
done

total=0
case "$mode" in
  --dry)
    say "DRY RUN. Nothing is verified and nothing is deleted."
    say ""; say "Group A -- deleted only if md5 matches LRZ:"
    for e in "${VERIFIED[@]}"; do
        l="${e%%|*}"; [ -e "$SRC/$l" ] && say "  $(human "$SRC/$l")  $l"
    done
    say ""; say "Group B -- deleted on their merits (no remote copy):"
    for e in "${UNVERIFIED[@]}"; do
        l="${e%%|*}"; [ -e "$SRC/$l" ] && { say "  $(human "$SRC/$l")  $l"; say "        ${e#*|}"; }
    done
    say ""; say "Run --check to verify Group A, then --go to delete."
    ;;

  --check|--go)
    [ "$mode" = "--go" ] && say "VERIFYING, then deleting what verifies." || say "VERIFYING only."
    ok=0; bad=0
    for e in "${VERIFIED[@]}"; do
        l="${e%%|*}"; r="${e#*|}"
        [ -e "$SRC/$l" ] || { say "  skip (already gone)  $l"; continue; }
        lm="$(md5 -q "$SRC/$l")"
        rm_="$(ssh "$REMOTE" "md5sum '$r' 2>/dev/null | cut -d' ' -f1")"
        if [ -n "$rm_" ] && [ "$lm" = "$rm_" ]; then
            say "  OK   $(human "$SRC/$l")  $l"
            ok=$((ok+1))
            if [ "$mode" = "--go" ]; then rm -f "$SRC/$l"; say "       deleted"; fi
        else
            say "  ** NO MATCH -- keeping $l"
            say "       local  $lm"
            say "       remote ${rm_:-<absent>}"
            bad=$((bad+1))
        fi
    done
    say ""; say "Group A: $ok verified, $bad refused."
    if [ "$mode" = "--go" ]; then
        say ""; say "Group B (no remote counterpart; deleted on their merits):"
        for e in "${UNVERIFIED[@]}"; do
            l="${e%%|*}"
            [ -e "$SRC/$l" ] || continue
            say "  removing $(human "$SRC/$l")  $l"
            rm -rf "${SRC:?}/$l"
        done
        say ""
        say "Done. Now re-check, because deleting frees nothing while a snapshot holds it:"
        say "    tmutil listlocalsnapshots /"
        say "    df -h /System/Volumes/Data"
    fi
    [ "$bad" -gt 0 ] && exit 1 || exit 0
    ;;
  *) say "usage: $0 [--dry|--check|--go]"; exit 2 ;;
esac
