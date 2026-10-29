#!/usr/bin/env bash
# Stage the five archives that version 2 of the Zenodo record adds (2026-10-29), next to the
# ones zenodo_upload.sbatch builds, so that
#
#     scripts/lrz/zenodo_stage_v2.sh                                  # login node, ~1 GB of I/O
#     DEPOSIT_ID=22935411 sbatch scripts/lrz/zenodo_upload.sbatch     # uploads what the draft lacks
#     scripts/lrz/zenodo_v2_finish.sh                                 # docs, metadata, drop the 2025 zip
#
# fills the draft. Nothing here publishes.
#
# Each archive is assembled from symlinks and tarred with -h, so the sources stay where they
# are. The functional inputs are checked against the sha256 recorded when the transfer runs
# were made (functional_2026-09-17/SUMMARY.md), so a changed file cannot slip in under its
# old name. An archive already staged is reused; delete it to rebuild.

set -euo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
ARCH="$DSS/plm_choice_archive_2026-10-08/internal_backup"   # md5-verified mirror of the Mac's analysis outputs
FUNC="$ARCH/functional_2026-09-17"
PROC="$DSS/plm_choice_data/data/processed"
STAGE="$DSS/zenodo_stage"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

BUILD="$(mktemp -d "$DSS/zenodo_v2_build.XXXX")"
trap 'rm -rf "$BUILD"' EXIT

# link <archive> <source>... : symlink each source into the archive's directory
link () {
    local dir="$BUILD/$1"; shift
    mkdir -p "$dir"
    for src in "$@"; do
        [ -e "$src" ] || { echo "missing: $src"; exit 1; }
        ln -s "$src" "$dir/"
    done
}

# alias <archive>/<name> <source> : symlink a source under another name
alias_as () {
    [ -e "$2" ] || { echo "missing: $2"; exit 1; }
    mkdir -p "$(dirname "$BUILD/$1")"
    ln -s "$2" "$BUILD/$1"
}

pack () {
    local name="$1"
    if [ -s "$STAGE/$name.tar" ]; then echo "reuse $name.tar"; return; fi
    echo "building $name.tar"
    tar -C "$BUILD" -h --sort=name --owner=0 --group=0 --numeric-owner \
        -cf "$STAGE/$name.tar.part" "$name"
    mv "$STAGE/$name.tar.part" "$STAGE/$name.tar"
}

check_sha () {   # check_sha <file> <first 12 hex of sha256>
    local got
    got="$(sha256sum "$1" | cut -c1-12)"
    [ "$got" = "$2" ] || { echo "sha256 mismatch: $1 ($got, expected $2)"; exit 1; }
}

check_sha "$ARCH/c1_strat_2026-09-15/ec_strat_v2_freeze.json" 176776877a6b
check_sha "$ARCH/c1_strat_2026-09-15/ec_labels_v2.tsv"         b2fbfa584a3d
check_sha "$FUNC/go_mf_cohort_freeze.json"                     9d5fa39db1b4
check_sha "$FUNC/go_mf_labels.tsv"                             20b220eea2b3
check_sha "$FUNC/union_allvsall.m8"                            c87beac6e77d
check_sha "$DSS/c1_strat_offload/functional/go-basic.obo"      a77e356737da

# 1. EC and GO-MF cohorts, the 1-nearest-neighbour transfer results and the MMseqs2 baseline.
#    The 26 embedding slices the runs read are left out: they are cut from the deposited .h5
#    files (and the untrained controls, which the code regenerates) by union_ids.txt.
link functional_annotation \
    "$ARCH/c1_strat_2026-09-15/ec_labels_v2.tsv" \
    "$ARCH/c1_strat_2026-09-15/ec_strat_v2_freeze.json" \
    "$FUNC/ec_v2_transfer" \
    "$FUNC/ec_tau_b_v2_euclidean.csv" \
    "$FUNC/go_mf_labels.tsv" \
    "$FUNC/go_mf_cohort_freeze.json" \
    "$DSS/c1_strat_offload/functional/go-basic.obo" \
    "$FUNC/go_mf_transfer" \
    "$FUNC/go_mf_transfer_noPB" \
    "$FUNC/union.fasta" \
    "$FUNC/union_ids.txt" \
    "$FUNC/union_allvsall.m8" \
    "$FUNC/union_allvsall.summary.json"
pack functional_annotation

# 2. Test pairs rescored from experimental structures (US-align), and the three target
#    tables the probes were scored on.
link experimental_tm \
    "$PROC/exp_tm/sets/exptm_matched.parquet" \
    "$PROC/exp_tm/sets/exptm_wholechain.parquet" \
    "$PROC/exp_tm/sets/exptm_afdbtarget.parquet" \
    "$ARCH/experimental_tm_2026-09-18/tmscore_exp.parquet"
pack experimental_tm

# 3. Per-model distance histograms: random pairs (Figure 2) and aligner-found pairs
#    (Table S4, Figure S6).
alias_as distance_histograms/random_pairs  "$ARCH/final_figures_2026-09-20/ridge/summaries"
alias_as distance_histograms/aligner_pairs "$ARCH/ridge_full_2026-09-19/summaries"
pack distance_histograms

# 4. The 5,000,000 random pairs and their distances under every model (Figures 2 and 3).
link random_pairs \
    "$DSS/random_pairs/random_pairs_distances.parquet" \
    "$DSS/random_pairs/summary.json" \
    "$DSS/random_pairs/convergence.csv"
pack random_pairs

# 5. New2024: its pairs, the 5 proteins excluded as public before the newest training data,
#    and the embeddings of its proteins.
link new2024 \
    "$PROC/2024_new_e1/sets/new2024.parquet" \
    "$REPO/docs/zenodo/new2024_exclude_public_before_training.txt"
alias_as new2024/embeddings "$PROC/2024_new_e1/embeddings"
pack new2024

echo
for name in functional_annotation experimental_tm distance_histograms random_pairs new2024; do
    printf '  %-28s %14s bytes\n' "$name.tar" "$(stat -c%s "$STAGE/$name.tar" | numfmt --grouping)"
done
echo "staged in $STAGE; next: DEPOSIT_ID=22935411 sbatch scripts/lrz/zenodo_upload.sbatch"
