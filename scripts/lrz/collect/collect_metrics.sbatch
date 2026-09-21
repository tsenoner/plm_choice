#!/bin/bash
# Re-collect the probe grid into one CSV.
#
# Runs as a batch job, not on the login node: the walk is pure GPFS metadata work and
# a login-node run wedged in cxiWaitEventWait for 35 minutes having issued 684 reads
# and zero writes. Compute nodes have their own GPFS client, and a batch job is not
# subject to the login-node watchdog.
#
#     sbatch ~/c1_strat/collect_metrics.sbatch
#SBATCH -J collect-metrics
#SBATCH -p lrz-cpu
#SBATCH --qos=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -o /dss/dsshome1/02/ge45ted2/plm_choice/logs/%x-%j.out
#SBATCH -e /dss/dsshome1/02/ge45ted2/plm_choice/logs/%x-%j.err

set -euo pipefail
unset VIRTUAL_ENV

echo "host=$(hostname) start=$(date -Is)"
~/plm_choice/.venv/bin/python ~/c1_strat/collect_probe_metrics.py ~/c1_strat/probe_metrics.csv
echo "end=$(date -Is)"
ls -la ~/c1_strat/probe_metrics.csv
