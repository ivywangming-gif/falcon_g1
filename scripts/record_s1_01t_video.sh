#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/root/autodl-tmp/robotics/falcon-g1-access-push"
FALCON_UPSTREAM="/root/autodl-tmp/robotics/falcon_sandbox/FALCON"
FALCON_ENV="/root/autodl-tmp/conda/envs/falcon_sim2sim"
AGILE="/root/autodl-tmp/robotics/projects/g1_access_push"
VIDEO_ROOT="/root/autodl-tmp/falcon_videos"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="s1_01t_chest_stand_video_${TS}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_NAME}"
VIDEO_PATH="${VIDEO_ROOT}/${RUN_NAME}.mp4"
SESSION="falcon_video_${TS}"
STATUS="${RUN_ROOT}/status.env"

mkdir -p "$RUN_ROOT" "$VIDEO_ROOT"

if pgrep -af \
    'video_simulator.py|s1_01t_policy_60s.py|headless_simulator.py|headless_policy.py' \
    | grep -v grep >/dev/null; then

    echo "ERROR=ANOTHER_FALCON_TEST_IS_ACTIVE"
    pgrep -af \
        'video_simulator.py|s1_01t_policy_60s.py|headless_simulator.py|headless_policy.py'
    exit 30
fi

tmux new-session \
    -d \
    -s "$SESSION" \
    "env \
RUN_ROOT='$RUN_ROOT' \
VIDEO_PATH='$VIDEO_PATH' \
PROJECT_ROOT='$PROJECT_ROOT' \
FALCON_UPSTREAM='$FALCON_UPSTREAM' \
FALCON_ENV='$FALCON_ENV' \
AGILE='$AGILE' \
bash '$PROJECT_ROOT/scripts/video_worker.sh'"

tmux set-option -t "$SESSION" remain-on-exit on

echo "VIDEO_RUN_STARTED=YES"
echo "TMUX_SESSION=${SESSION}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "TARGET_VIDEO=${VIDEO_PATH}"

for _ in $(seq 1 240); do
    [ ! -f "$STATUS" ] || break

    if tmux display-message \
        -p \
        -t "$SESSION" \
        '#{pane_dead}' 2>/dev/null |
        grep -qx 1; then
        break
    fi

    sleep 1
done

echo
echo "===================== FINAL STATUS ====================="
cat "$STATUS" 2>/dev/null || echo "STATUS_FILE_MISSING"

echo
echo "===================== VIDEO REPORT ====================="
cat "${RUN_ROOT}/video_simulator_report.json" 2>/dev/null || true

echo
echo "===================== POLICY LOG TAIL =================="
tail -n 40 "${RUN_ROOT}/policy.log" 2>/dev/null || true

echo
echo "===================== SIMULATOR LOG TAIL ==============="
tail -n 40 "${RUN_ROOT}/simulator.log" 2>/dev/null || true

echo
echo "DOWNLOAD_VIDEO=${VIDEO_PATH}"
