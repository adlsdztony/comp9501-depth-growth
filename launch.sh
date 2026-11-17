#!/usr/bin/env bash
# Start only when invoked explicitly. No network, no access outside the run
# copy except symlinked bundled reference and the caller-supplied Python env.
set -euo pipefail
[[ $# -ge 4 ]] || { echo "usage: bash launch.sh MODE PYTHON CPU_LIST NEW_RUN_DIR [GPU_UUID [SEED]]" >&2; exit 2; }
mode=$1; py=$2; cpus=$3; run=$4
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=$(realpath -s -- "$py")
[[ -x "$py" && ! -e "$run" && ! -L "$run" ]] || { echo "Need executable Python and a new run path" >&2; exit 2; }
taskset -c "$cpus" "$py" -c 'import os; assert len(os.sched_getaffinity(0)) == 12, "select exactly 12 accessible logical CPUs"'
case "$mode" in
  smoke) [[ $# -eq 4 ]]; src=full_mt10; uuid=cpu; seed=17711 ;;
  pilot) [[ $# -eq 5 ]]; src=pilot_3task; uuid=$5; seed=17701 ;;
  full) [[ $# -ge 5 && $# -le 6 ]]; src=full_mt10; uuid=$5; seed=${6:-17711} ;;
  *) echo "MODE must be smoke, pilot or full" >&2; exit 2 ;;
esac
if [[ "$mode" == full ]]; then
  [[ "$seed" =~ ^1771[123]$ ]] || { echo "Full seed must be 17711, 17712 or 17713" >&2; exit 2; }
fi
if [[ "$uuid" != cpu ]]; then
  [[ "$uuid" =~ ^GPU-[0-9a-fA-F-]+$ ]]
  nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -Fx -- "$uuid" >/dev/null
  export CUDA_VISIBLE_DEVICES="$uuid"
else
  export CUDA_VISIBLE_DEVICES=
fi
mkdir -- "$run"
run=$(realpath -- "$run")
cp -a "$script_root/$src/." "$run/"
export WANDB_MODE=disabled PYTHONNOUSERSITE=1 MPLBACKEND=Agg MUJOCO_GL=egl
export TMPDIR=$(mktemp -d /tmp/mt10.XXXXXXXX)
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT
cd -- "$run"
if [[ "$mode" == pilot ]]; then
  mkdir reference
  ln -s -- "$script_root/full_mt10/reference/TOPPO_author" reference/TOPPO_author
  "$py" -c 'import json,sys; print(json.dumps({"verified_gpu_uuids":[sys.argv[1]]}))' "$uuid" >status.json
  taskset -c "$cpus" "$py" -B -u orchestrator.py
elif [[ "$mode" == smoke ]]; then
  taskset -c "$cpus" "$py" -B -u orchestrator.py --seed "$seed" --gpu-uuid cpu --cpu-smoke
else
  taskset -c "$cpus" "$py" -B -u orchestrator.py --seed "$seed" --gpu-uuid "$uuid" --wall-seconds 5300 --reserve-gib 45
fi
