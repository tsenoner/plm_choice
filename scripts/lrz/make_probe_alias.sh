#!/bin/bash
# Make a DATASET alias for probe_grid.sbatch: a directory of symlinks onto an existing
# dataset's splits and a chosen subset of its embedding arms. Same data, new name, so the
# cells land in models/<alias>/... and cannot resume from or overwrite the source's cells.
# This is how full100_p10 was made from sprot_pre2024_e1 (sets) + embeddings_cohort2k.
#
#     scripts/lrz/make_probe_alias.sh full100_p10 full100_p10_h256 \
#         ankh_large esm2_3b esm2_8m esmc_600m prott5 prottucker
#     scripts/lrz/make_probe_alias.sh full100_p10 full100_p10_lindist     # every arm
#
# Refuses to touch an alias that already exists: a changed arm set would silently re-map
# probe_grid.sbatch's array indices onto different arms.
set -euo pipefail

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
P="$DSS/plm_choice_data/data/processed"
SRC_NAME=${1:?source dataset, e.g. full100_p10}
ALIAS=${2:?alias name, e.g. full100_p10_h256}
shift 2
SRC="$P/$SRC_NAME"
DST="$P/$ALIAS"

[ -d "$SRC/embeddings" ] || { echo "no $SRC/embeddings"; exit 2; }
[ -e "$DST" ] && { echo "$DST exists; not touching it"; exit 2; }

if [ "$#" -eq 0 ]; then
    mapfile -t ARMS < <(find -L "$SRC/embeddings" -maxdepth 1 -name '*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
else
    ARMS=("$@")
fi

mkdir -p "$DST/embeddings"
ln -s "$(readlink -f "$SRC/sets")" "$DST/sets"
for arm in "${ARMS[@]}"; do
    target=$(readlink -f "$SRC/embeddings/$arm.h5")
    [ -f "$target" ] || { echo "missing arm $arm in $SRC"; rm -rf "$DST"; exit 2; }
    ln -s "$target" "$DST/embeddings/$arm.h5"
done
echo "$DST: ${#ARMS[@]} arms"
find "$DST/embeddings" -name '*.h5' -printf '%f\n' | sort | nl -v0
