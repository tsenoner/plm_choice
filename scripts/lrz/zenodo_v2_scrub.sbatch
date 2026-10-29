#!/bin/bash
# Rewrite the staged Zenodo v2 archives into their public form (zenodo_v2_scrub.py), then prove
# one rewritten probe still works end to end, and scan the 15 embedding files for metadata.
#
#     sbatch scripts/lrz/zenodo_v2_scrub.sbatch            # from the repo root on the cluster
#     ONLY="pair_table_counts.tar" sbatch scripts/lrz/zenodo_v2_scrub.sbatch
#
# Output: $STAGE/clean/<published name>.tar plus a .md5 each. The staged originals are read,
# never changed. Upload afterwards with zenodo_v2_replace.sbatch.
#
#SBATCH -J zenodo-scrub
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
PY="$REPO/.venv/bin/python"
DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
STAGE="$DSS/zenodo_stage"
PROC="$DSS/plm_choice_data/data/processed"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# 1. Rewrite and check every archive.
# shellcheck disable=SC2086
"$PY" "$REPO/scripts/lrz/zenodo_v2_scrub.py" --stage "$STAGE" ${ONLY:+--only $ONLY}

# 2. End to end: unpack one rewritten cell next to the data its hparams.yaml now names (the
#    deposited test pairs and embedding file, laid out as the record's README tells users),
#    delete its cached predictions, recompute them with the repo's evaluate.py, and compare
#    with the deposited ones. evaluate.py swallows its own errors and exits 0, so deleting the
#    cache first is what makes a silent failure visible.
if [ -s "$STAGE/clean/probes_native.tar" ]; then
    T="$(mktemp -d "$DSS/zenodo_e2e.XXXX")"
    trap 'rm -rf "$T"' EXIT
    cd "$T"
    mkdir -p swissprot_pairs/embeddings deposited
    ln -s "$PROC/sprot_pre2024_e1/sets" swissprot_pairs/sets
    ln -s "$PROC/sprot_pre2024/embeddings_cohort2k/esm2_8m.h5" swissprot_pairs/embeddings/esm2_8m.h5
    cell=probes_native/fnn/fident/esm2_8m
    tar -xf "$STAGE/clean/probes_native.tar" "$cell"
    mv "$cell"/evaluation_results/*_predictions_targets.npz "$cell"/evaluation_results/*_metrics.txt deposited/
    "$PY" "$REPO/src/evaluation/evaluate.py" --run_dir "$cell" --n_bootstrap 1 --force_recompute
    "$PY" - "$cell" <<'PY'
import glob, re, sys
import numpy as np
cell = sys.argv[1]
new = glob.glob(f"{cell}/evaluation_results/*_predictions_targets.npz")
old = glob.glob("deposited/*_predictions_targets.npz")
if len(new) != 1 or len(old) != 1:
    sys.exit(f"FAIL: recomputed predictions missing ({new})")
a, b = np.load(old[0]), np.load(new[0])
if a["targets"].shape != b["targets"].shape or not np.array_equal(a["targets"], b["targets"]):
    sys.exit("FAIL: targets differ from the deposited ones")
d = float(np.max(np.abs(a["predictions"] - b["predictions"])))
def rho(path):
    return float(re.search(r"^Spearman: (\S+)", open(path).read(), re.M).group(1))
r_old = rho(glob.glob("deposited/*_metrics.txt")[0])
r_new = rho(glob.glob(f"{cell}/evaluation_results/*_metrics.txt")[0])
print(f"end-to-end {cell}: {len(a['targets']):,} pairs, max |prediction diff| {d:.2e}, "
      f"Spearman deposited {r_old:.6f} recomputed {r_new:.6f}")
if d > 1e-4 or abs(r_old - r_new) > 1e-5:
    sys.exit("FAIL: recomputed predictions do not match the deposited ones")
print("end-to-end check ok")
PY
    cd "$REPO"
fi

# 3. The 15 embedding files: no attributes, no groups, every name an accession.
"$PY" - "$PROC/sprot_pre2024/embeddings_cohort2k" <<'PY'
import glob, os, re, sys
import h5py
acc = re.compile(r"^[A-Z0-9]{6,10}$")
bad = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.h5"))):
    with h5py.File(f, "r") as h:
        names = list(h.keys())
        groups = sum(isinstance(h[n], h5py.Group) for n in names[:2000])
        sample_attrs = sum(len(h[n].attrs) for n in names[::max(1, len(names) // 2000)])
        odd = [n for n in names if not acc.match(n)]
        ok = not h.attrs and not groups and not sample_attrs and not odd
        bad += not ok
        print(f"  {os.path.basename(f):16s} {len(names):>7,} datasets  file attrs {len(h.attrs)}  "
              f"groups {groups}  sampled dataset attrs {sample_attrs}  non-accession names {len(odd)}"
              f"{'' if ok else '  <-- CHECK'}")
sys.exit(1 if bad else 0)
PY
echo "scrub job done"
