#!/bin/bash
# Functional arm read-out: leave-one-out 1-NN annotation transfer for EC and GO-MF, against the
# MMseqs2 homology baseline, over all 26 arms (15 pretrained + 11 random-init).
#
#     cd ~/plm_choice && mkdir -p logs
#     sbatch --array=0-2 scripts/lrz/functional_transfer.sbatch     # ec_v2 | go_mf | go_mf_noPB
#
# ---------------------------------------------------------------------------
# WHY THIS REPLACES tau_b AS THE PRIMARY READ-OUT. On the v2 EC cohort, Kendall tau_b over all
# 23,815,351 pairs runs 0.0005-0.064, while leave-one-out 1-NN exact-EC transfer on the SAME
# distances runs 0.534-0.809 against a chance rate of 0.0022. 99.78% of pairs share no EC and
# 77.0% differ at the class level, so a global rank correlation is carried by pairs whose distance
# ordering has no functional meaning. tau_b stays as a secondary number (--no-tau turns it off).
#
# THE COMPARISON THAT ANSWERS THE REVIEWERS is not chance, it is homology: the same transfer done
# from the best MMseqs2 hit (hbi_evalue / hbi_fident rows). Plus the identity strata -- all
# neighbours, neighbours <30% identity, neighbours with no hit at E<=1e-3 -- which is a
# stratification, not a gate, so report the eligible-query count per stratum.
#
# THIS RUNS ON THE CLUSTER ON PURPOSE. It was first run on the Mac and died with the session; the
# inputs (2.5 GB of slices) already live on GPFS, so the durable place for it is a batch job.
# The code is on branch feat/functional-transfer, checked out at ~/plm_choice_go via git worktree
# (NOT scp'd) and imported with PYTHONPATH so the venv at ~/plm_choice/.venv is reused.
#
# Memory: the GO cohort is 20,073 proteins, so each distance matrix is 20,073^2 float64 = 3.2 GB
# and the label-score matrices are float32 of the same shape. 120G leaves room for two at once.
# ---------------------------------------------------------------------------
#SBATCH -J func-transfer
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu                    # MANDATORY: lrz-cpu rejects the default QOS
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV

WT="$HOME/plm_choice_go"
PY="$HOME/plm_choice/.venv/bin/python"
F="$HOME/c1_strat/functional"
EMB="$HOME/c1_strat/slices_union"
export PYTHONPATH="$WT/src"

[ -d "$WT/src/evaluation" ] || { echo "worktree missing: $WT"; exit 2; }
[ -d "$EMB" ] || { echo "slices missing: $EMB"; exit 2; }

IDX=${SLURM_ARRAY_TASK_ID:?set --array=0-2}
COMMON=(--emb-dir "$EMB" --distances euclidean cosine
        --identity-m8 "$F/union_allvsall.m8" --n-boot 2000 --seed 42)

case "$IDX" in
  0) ARGS=(--labels-kind ec --freeze "$HOME/c1_strat/ec_strat_v2_freeze.json"
           --labels "$HOME/c1_strat/ec_labels_v2.tsv" --out-dir "$F/out_ec_v2") ;;
  1) ARGS=(--labels-kind go --freeze "$F/go_mf_cohort_freeze.json"
           --labels "$F/go_mf_labels.tsv" --go-obo "$F/go-basic.obo"
           --out-dir "$F/out_go_mf") ;;
  2) ARGS=(--labels-kind go --freeze "$F/go_mf_cohort_freeze.json"
           --labels "$F/go_mf_labels.tsv" --go-obo "$F/go-basic.obo"
           --drop-protein-binding --out-dir "$F/out_go_mf_noPB") ;;
  *) echo "index $IDX out of range (0-2)"; exit 2 ;;
esac

echo "task=$IDX host=$(hostname) arms=$(find -L "$EMB" -name '*.h5' | wc -l)"
srun "$PY" -m evaluation.transfer_report "${COMMON[@]}" "${ARGS[@]}"

OUT=$(printf '%s\n' "${ARGS[@]}" | tail -1)
ls -la "$OUT"
echo "done: task $IDX"
