#!/bin/bash
# Probe grid on the post-rebuild 10% splits: one array task = one (target, embedding arm),
# each training the fnn probe and scoring it plus the training-free euclidean baseline.
#
#     cd ~/plm_choice && mkdir -p logs
#     # pretrained arms (15 x 3 targets = 45 tasks)
#     DATASET=sprot_pre2024_e1_sub10 EXPECT_ARMS=15 \
#         sbatch --array=0-44%10 scripts/lrz/probe_grid.sbatch
#     # untrained random-init arms on the MIG slices (11 x 3 = 33 tasks)
#     DATASET=sprot_pre2024_e1_sub10_randinit EXPECT_ARMS=11 \
#         sbatch -p lrz-dgx-a100-40x8-mig --qos=mig --gres=gpu:3g.20gb:1 \
#         --array=0-32%10 scripts/lrz/probe_grid.sbatch
#
# ---------------------------------------------------------------------------
# WHAT THIS REPRODUCES. The published protocol, on corrected data: fnn, hidden 64, lr 1e-4,
# batch 1024, early-stopping patience 5, max 100 epochs, seed 42, validation once per
# epoch (all 55 published subset logs record val_check_interval 1.0; run_experiments.py's
# 0.2 default would add +88% validation overhead and change when early stopping fires).
# The splits are a 10% positional sample (seed 42) of the post-E1 splits -- HFSP fixed,
# pairs deduplicated -- drawn by scripts/create_subset_datasets.py into a NEW cohort dir,
# so no completion marker or last.ckpt from a pre-fix run can be picked up (run_experiments
# would skip or resume those silently). sha256 of train/val/test:
#     c4a54110... / 7db7e727... / 25164587...
# The <=1022 cohort filter is applied at load time by shared/datasets.py.
#
# TWO DELIBERATE DEPARTURES, both to be stated in Methods (M-11):
#   * CUDA. train.py sets float32 matmul precision "medium" when CUDA is present; every
#     published run used full fp32 on Apple MPS. Accepted 2026-09-17.
#   * One seed (42) and 10% of pairs, as the approximation for the first draft; the 100%
#     retrain follows.
#
# WHY THE EMBEDDING IS STAGED TO NODE-LOCAL DISK. The loader reads one HDF5 dataset per
# protein per pair. On GPFS that metadata-heavy access pattern measured ~20x slower than
# local disk (B4: 86 s locally vs ~27 min per arm). The arm is copied to local disk when
# there is room, and the run falls back to DSS otherwise; the log says which.
#
# WANDB. The account has no wandb credentials on LRZ, so runs are offline. evaluate.py
# reads hparams back from wandb/*/files/config.yaml and falls back to hparams.yaml; we write
# hparams.yaml ourselves so that fallback always exists. evaluate.py also catches its own
# exceptions and exits 0, so success is asserted on the metrics file instead.
# ---------------------------------------------------------------------------
#SBATCH -J probe-grid
#SBATCH -p lrz-hgx-a100-80x4,lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-v100x2,lrz-dgx-1-v100x8
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1                 # MANDATORY: without it the job sits PD forever
#SBATCH --time=12:00:00              # MANDATORY: default is 1h, QOS caps at 2 days
#SBATCH --cpus-per-task=12           # 10 DataLoader workers + main process
#SBATCH --mem=64G
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail
unset VIRTUAL_ENV

DSS=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2
DATASET="${DATASET:?set DATASET, e.g. sprot_pre2024_e1_sub10}"
EXPECT_ARMS="${EXPECT_ARMS:?set EXPECT_ARMS (the embedding count the dataset must hold)}"
SRC="$DSS/plm_choice_data/data/processed/$DATASET"
PARAMS=(fident alntmscore hfsp)
# Early stopping: 5 is the published protocol. Raise it to re-test a cell that stopped
# suspiciously early -- three alntmscore cells peaked at epochs 2-4 while their neighbours
# trained for 20-47, which is the signature of an early plateau rather than a converged fit.
PATIENCE="${PATIENCE:-5}"

cd "$HOME/plm_choice"
PY=./.venv/bin/python

mapfile -t ARMS < <(find -L "$SRC/embeddings" -maxdepth 1 -name '*.h5' -printf '%f\n' | sed 's/\.h5$//' | sort)
N=${#ARMS[@]}
if [ "$N" -ne "$EXPECT_ARMS" ]; then
    echo "found $N embedding arms in $SRC/embeddings, expected $EXPECT_ARMS"; exit 2
fi
IDX=${SLURM_ARRAY_TASK_ID:?set --array}
if [ "$IDX" -ge $((N * ${#PARAMS[@]})) ]; then echo "index $IDX out of range"; exit 2; fi
PARAM=${PARAMS[$((IDX / N))]}
ARM=${ARMS[$((IDX % N))]}

# Stage the arm off GPFS. The loader reads one HDF5 dataset per protein per pair, and that
# metadata-heavy pattern measured ~100x slower on GPFS than on a staged copy (0.20-0.24 it/s
# against 23 it/s). Two 12 h walltimes were burned on esm2_3b because a free-space PREDICTION
# refused to stage it -- the second time by 87 MB. So stop predicting: try RAM, then local
# disk, and fall back to GPFS only when a copy actually fails.
STAGE_ROOT="${TMPDIR:-/tmp}/probe_${SLURM_JOB_ID}_${IDX}"
H5="$SRC/embeddings/$ARM.h5"
DATA="$SRC"
for CANDIDATE in "/dev/shm/probe_${SLURM_JOB_ID}_${IDX}" "$STAGE_ROOT"; do
    TRY="$CANDIDATE/$DATASET"
    if mkdir -p "$TRY/embeddings" 2>/dev/null && cp -L "$H5" "$TRY/embeddings/$ARM.h5" 2>/dev/null; then
        ln -sfn "$(readlink -f "$SRC/sets")" "$TRY/sets"
        DATA="$TRY"
        STAGE_ROOT="$CANDIDATE"
        trap 'rm -rf "$STAGE_ROOT"' EXIT
        echo "staged $ARM.h5 to $DATA"
        break
    fi
    rm -rf "$CANDIDATE" 2>/dev/null || true
done
[ "$DATA" = "$SRC" ] && echo "NOT staged (both /dev/shm and local disk refused the copy); reading from DSS"

export WANDB_MODE=offline WANDB_SILENT=true
echo "task=$IDX param=$PARAM arm=$ARM dataset=$DATASET host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_cell () {
    local model=$1
    local exp="$HOME/plm_choice/models/$DATASET/$model/$PARAM/$ARM"
    local extra=()
    if [ -f "$exp/training_complete.txt" ]; then
        echo "[$model] training already complete, going straight to evaluation"
    else
        [ -f "$exp/checkpoints/last.ckpt" ] && extra=(--resume_from_checkpoint)
        mkdir -p "$exp"
        $PY src/training/train.py \
            --model_type "$model" --param_name "$PARAM" \
            --embedding_file "$DATA/embeddings/$ARM.h5" \
            --data_dir "$DATA/sets" --output_base_dir "$exp" \
            --num_workers 10 --batch_size 1024 --learning_rate 0.0001 \
            --max_epochs 100 --early_stopping_patience "$PATIENCE" --val_check_interval 1.0 \
            --seed 42 --wandb_project which-plm ${extra[@]+"${extra[@]}"}
    fi
    # evaluate.py reads hparams back from wandb/*/files/config.yaml, which records the
    # staging path of whichever job trained the cell. That path is gone on any later node,
    # so a re-run of evaluation alone (training_complete.txt present, no new wandb run)
    # would resolve data_dir/embedding_file to a dangling symlink and fail -- which is how
    # 5795274_16 died after a completed 1:54 h training run. Repoint the config at this
    # job's paths before evaluating; idempotent, and it keeps the rest of the config.
    $PY - "$exp" "$DATA" "$ARM" <<'PYEOF'
import sys, pathlib, yaml
exp, data, arm = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for cfg in exp.glob("wandb/*/files/config.yaml"):
    try:
        c = yaml.safe_load(cfg.read_text()) or {}
    except yaml.YAMLError as e:
        print(f"skip {cfg}: {e}"); continue
    changed = False
    for key, value in (("data_dir", f"{data}/sets"), ("embedding_file", f"{data}/embeddings/{arm}.h5")):
        if isinstance(c.get(key), dict) and c[key].get("value") != value:
            c[key]["value"] = value; changed = True
    if changed:
        cfg.write_text(yaml.safe_dump(c)); print(f"repointed {cfg}")
PYEOF
    cat > "$exp/hparams.yaml" <<EOF
model_type: $model
param_name: $PARAM
embedding_file: $DATA/embeddings/$ARM.h5
data_dir: $DATA/sets
batch_size: 1024
seed: 42
dataset: $DATASET
EOF
    $PY src/evaluation/evaluate.py --run_dir "$exp" --n_bootstrap 1000 --force_recompute
    if ! ls "$exp"/evaluation_results/*_metrics.txt >/dev/null 2>&1; then
        echo "[$model] evaluation produced no metrics file"; return 1
    fi
    echo "[$model] done: $(ls "$exp"/evaluation_results/*_metrics.txt)"
}

run_cell fnn
run_cell euclidean
echo "done: $PARAM / $ARM"
