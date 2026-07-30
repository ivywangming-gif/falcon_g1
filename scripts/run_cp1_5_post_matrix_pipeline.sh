#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/root/autodl-tmp/robotics/falcon-g1-access-push
PYTHON=/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python
TIMESTAMP=20260730_105900
export PYTHONPATH="$REPO/src"
export XDG_CACHE_HOME="$REPO/.cache/xdg"
export PIP_CACHE_DIR="$REPO/.cache/pip"
export TMPDIR="$REPO/.cache/tmp"
cd "$REPO"

while pgrep -f '^/root/autodl-tmp/conda/envs/falcon_isaaclab/bin/python scripts/run_cp1_5_constant_matrix.py( |$)' >/dev/null; do
    echo "WAITING_FOR_CONSTANT_MATRIX $(date -u +%FT%TZ)"
    sleep 10
done

echo "START_PUSH_READY $(date -u +%FT%TZ)"
"$PYTHON" scripts/run_cp1_5_push_ready_matrix.py \
    --campaign-root "$REPO/runs/cp1_5_push_ready_$TIMESTAMP" --timestamp "$TIMESTAMP"

echo "START_EXTERNAL_LOADS $(date -u +%FT%TZ)"
"$PYTHON" scripts/run_cp1_5_external_loads.py \
    --campaign-root "$REPO/runs/cp1_5_external_loads_$TIMESTAMP" --timestamp "$TIMESTAMP"

echo "START_OFFICIAL_SIM2SIM $(date -u +%FT%TZ)"
SIM_SUMMARY="$REPO/runs/cp1_5_sim2sim_$TIMESTAMP/summary.json"
if /root/autodl-tmp/conda/envs/falcon_sim2sim/bin/python \
    scripts/run_cp1_5_sim2sim_comparison.py \
    --campaign-root "$REPO/runs/cp1_5_sim2sim_$TIMESTAMP" --timestamp "$TIMESTAMP"; then
    "$PYTHON" scripts/finalize_cp1_5_reports.py \
        --constant-summary "$REPO/artifacts/cp1_5/constant_command_summary.json" \
        --sim2sim-summary "$SIM_SUMMARY"
else
    echo "OFFICIAL_SIM2SIM_ADAPTER_FAILED_WITHOUT_UPSTREAM_CHANGE"
    "$PYTHON" scripts/finalize_cp1_5_reports.py \
        --constant-summary "$REPO/artifacts/cp1_5/constant_command_summary.json"
fi

echo "START_VIDEO_FINALIZATION $(date -u +%FT%TZ)"
"$PYTHON" scripts/finalize_cp1_5_videos.py --timestamp "$TIMESTAMP"
echo "POST_MATRIX_PIPELINE_COMPLETE $(date -u +%FT%TZ)"
