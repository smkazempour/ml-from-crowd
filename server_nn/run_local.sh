#!/usr/bin/env bash
# Invoke from an interactive allocation or a dedicated host, never a login node.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${BUNDLE:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-runs}"
WORKERS="${WORKERS:-24}"
THREADS="${THREADS:-2}"
MEMORY_PER_WORKER_GIB="${MEMORY_PER_WORKER_GIB:-6}"
RESERVE_GIB="${RESERVE_GIB:-24}"
PHASES="${PHASES:-nn3 depth}"
FRESH="${FRESH:-0}"

cd -- "$BUNDLE"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

check_args=()
train_args=()
if [[ "$FRESH" == "1" ]]; then
    check_args+=(--allow-runtime-difference)
    train_args+=(--no-reuse-local)
elif [[ "$FRESH" != "0" ]]; then
    printf '%s\n' 'FRESH must be 0 or 1' >&2
    exit 2
fi

"$PYTHON" -B -m server_nn.check_environment --workers "$WORKERS" --threads "$THREADS" \
    --memory-per-worker-gib "$MEMORY_PER_WORKER_GIB" --reserve-gib "$RESERVE_GIB" "${check_args[@]}"
"$PYTHON" -B -m server_nn.verify --bundle .
if [[ "$FRESH" == "0" ]]; then
    "$PYTHON" -B -m server_nn.verify --bundle . --bridge --threads "$THREADS"
fi

# Deliberate whitespace separation: PHASES contains only validated phase names.
read -r -a selected_phases <<< "$PHASES"
if (( ${#selected_phases[@]} == 0 )); then
    printf '%s\n' 'PHASES must contain at least one of nn3, depth, width' >&2
    exit 2
fi
for phase in "${selected_phases[@]}"; do
    case "$phase" in nn3|depth|width) ;; *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2 ;; esac
done
for phase in "${selected_phases[@]}"; do
    "$PYTHON" -B -u -m server_nn.train --bundle . --output "$OUTPUT" --phase "$phase" \
        --workers "$WORKERS" --threads "$THREADS" \
        --memory-per-worker-gib "$MEMORY_PER_WORKER_GIB" --reserve-gib "$RESERVE_GIB" "${train_args[@]}"
    "$PYTHON" -B -u -m server_nn.evaluate --bundle . --output "$OUTPUT" --phase "$phase"
done
