#!/usr/bin/env bash
# Status of a probe-grid array on LRZ, in one screen.
#
#     scripts/lrz/grid_status.sh            # the array that is queued now
#     scripts/lrz/grid_status.sh 5817287    # a specific array
#
# Nothing is hardcoded: the job id comes from the queue and the dataset name is read out
# of the job's own log, so this keeps working after the current grid is replaced.
# Override the ssh host with LRZ_HOST=... if it is not "ai".
set -uo pipefail

ssh "${LRZ_HOST:-ai}" bash -s -- "${1:-}" <<'REMOTE_EOF'
set -uo pipefail
JOB="${1:-}"
P="$HOME/plm_choice"
DSS_PROC=/dss/dssfs05/lwp-dss-0003/pr63ci/pr63ci-dss-0003/ge45ted2/plm_choice_data/data/processed

# Prefer a queued array. With nothing queued, pick the most recent FULL grid rather than
# the newest job id: a one-cell repair submission is often newer than the grid it repairs,
# and picking it reports the wrong dataset at 90/90 as though the grid were the subject.
# "Full" means the most array tasks; ties go to the newer job.
RECENT="$(sacct -u "$USER" -n -X --name=probe-grid -S now-21days -o JobID --parsable2 2>/dev/null \
            | sed 's/_.*//;s/ //g' | grep -E '^[0-9]+$' | sort -un)"
[ -z "$JOB" ] && JOB="$(squeue -u "$USER" -h -n probe-grid -o '%A' 2>/dev/null | sort -u | head -1)"
if [ -z "$JOB" ]; then
    best=""; best_n=0
    for j in $RECENT; do
        n="$(sacct -j "$j" -n -X -o JobID --parsable2 2>/dev/null | grep -c '_')"
        if [ "$n" -ge "$best_n" ]; then best_n="$n"; best="$j"; fi
    done
    JOB="$best"
fi
if [ -z "$JOB" ]; then echo "no probe-grid job found"; exit 1; fi

# The dataset the job was launched with, taken from its own log rather than assumed.
DATASET=""
for f in "$P"/logs/probe-grid-"$JOB"_*.out; do
    [ -e "$f" ] || continue
    DATASET="$(head -c 100000 "$f" 2>/dev/null | grep -ho 'dataset=[a-z0-9_]*' | head -1 | cut -d= -f2)"
    [ -n "$DATASET" ] && break
done
DATASET="${DATASET:-unknown}"

have=0
[ -d "$P/models/$DATASET" ] && have="$(find "$P/models/$DATASET" -name '*_metrics.txt' 2>/dev/null | wc -l)"

# The grid's size is 2 read-outs x 3 targets x (however many arms the dataset has), so it is
# NOT always 90: the random-init control has 11 arms and completes at 66. Count the arms in
# the dataset rather than assuming, or a finished grid reports as 73% done.
WANT=90
for base in "$DSS_PROC/$DATASET" "$P/models/$DATASET"; do
    if [ -d "$base/embeddings" ]; then
        a="$(find -L "$base/embeddings" -maxdepth 1 -name '*.h5' 2>/dev/null | wc -l)"
        [ "$a" -gt 0 ] && WANT=$((a * 3 * 2)) && break
    fi
done
# Fall back to the model tree: arms are the leaf dirs under one read-out/target pair.
if [ "$WANT" -eq 90 ] && [ -d "$P/models/$DATASET/fnn/fident" ]; then
    a="$(find "$P/models/$DATASET/fnn/fident" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    [ "$a" -gt 0 ] && WANT=$((a * 3 * 2))
fi
run="$(squeue -j "$JOB" -h -t RUNNING -r 2>/dev/null | wc -l)"
pend="$(squeue -j "$JOB" -h -t PENDING -r 2>/dev/null | wc -l)"
# Only the head of each log: the cohort filter runs before training, and a training log
# grows to ~30 MB of progress-bar output that grep would otherwise read in full on every
# file that does NOT match -- which is all of them when things are healthy.
warn=0
for f in "$P"/logs/probe-grid-"$JOB"_*.out; do
    [ -e "$f" ] || continue
    if head -c 2000000 "$f" 2>/dev/null | grep -q 'cohort WARNING'; then
        warn=$((warn + 1))
    fi
done

# Tasks that need resubmitting, as a comma list ready to paste into --array=.
# A task that timed out and was then re-run successfully needs nothing, so a bad sacct
# state is only a candidate: the deciding test is whether its cell has its metrics. The
# index -> (target, arm) mapping is read from the job's own "task=N param=T arm=A" line,
# so nothing here assumes an ordering that the sbatch could change.
bad=""
for idx in $(sacct -j "$JOB" -n -X -o JobID,State --parsable2 2>/dev/null \
              | awk -F'|' '$2 ~ /FAILED|TIMEOUT|OUT_OF_ME|CANCELLED/ {split($1,a,"_"); print a[2]}' \
              | grep -E '^[0-9]+$' | sort -un); do
    log="$P/logs/probe-grid-${JOB}_${idx}.out"
    [ -e "$log" ] || { bad="$bad,$idx"; continue; }
    line="$(head -c 100000 "$log" 2>/dev/null | grep -ho 'param=[a-z]* arm=[a-z0-9_]*' | head -1)"
    tgt="${line%% *}"; tgt="${tgt#param=}"
    arm="${line##* }"; arm="${arm#arm=}"
    if [ -z "$tgt" ] || [ -z "$arm" ]; then bad="$bad,$idx"; continue; fi
    n=0
    for ro in fnn euclidean; do
        d="$P/models/$DATASET/$ro/$tgt/$arm"
        [ -d "$d" ] && n=$((n + $(find "$d" -name '*_metrics.txt' 2>/dev/null | wc -l)))
    done
    [ "$n" -lt 2 ] && bad="$bad,$idx"
done
bad="${bad#,}"

pct=$(( have * 100 / WANT ))
filled=$(( pct / 5 ))
bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((20 - filled)) '')"

others="$(printf '%s\n' $RECENT | grep -v "^${JOB}$" | tail -4 | tr '\n' ' ')"
printf '\nprobe-grid %s   dataset %s\n\n' "$JOB" "$DATASET"
printf '  metrics   %2d/%d  [%s] %d%%\n' "$have" "$WANT" "$bar" "$pct"
printf '  tasks     running %-3s pending %-3s\n' "$run" "$pend"
if [ "$warn" -eq 0 ]; then
    printf '  cohort    0 warnings  OK\n'
else
    printf '  cohort    %s WARNING(S) -- an arm is off the shared cohort; do NOT collect\n' "$warn"
fi
if [ -n "$bad" ]; then
    printf '  resubmit  --array=%s\n' "$bad"
    printf '            (they resume from last.ckpt; PATIENCE=10 must match)\n'
else
    printf '  resubmit  nothing\n'
fi
if [ "$have" -ge "$WANT" ]; then
    printf '\n  GRID COMPLETE -- collect metrics in a Slurm job, then rebuild the figures.\n'
fi
[ -n "$others" ] && printf '\n  newest other arrays: %s(pass one as an argument)\n' "$others"
echo
REMOTE_EOF
