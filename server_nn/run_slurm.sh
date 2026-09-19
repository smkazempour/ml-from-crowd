#!/usr/bin/env bash
# Edit/override allocation and wall time to match your site's policy.
#SBATCH --job-name=stocktwits-nn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --output=stocktwits-nn-%j.log
#SBATCH --error=stocktwits-nn-%j.log

set -euo pipefail
: "${BUNDLE:?Set BUNDLE to the absolute transferred bundle directory before sbatch}"
: "${PYTHON:?Set PYTHON to the absolute virtual-environment Python before sbatch}"
cd -- "$BUNDLE"
# run_local performs allocation/memory checks, verification, training and scoring.
exec bash "$BUNDLE/server_nn/run_local.sh"
