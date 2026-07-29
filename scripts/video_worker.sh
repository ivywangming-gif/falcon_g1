#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

: "${RUN_ROOT:?}"
: "${VIDEO_PATH:?}"
: "${PROJECT_ROOT:?}"
: "${FALCON_UPSTREAM:?}"
: "${FALCON_ENV:?}"
: "${AGILE:?}"

SIM2REAL="${FALCON_UPSTREAM}/sim2real"
STATUS="${RUN_ROOT}/status.env"
SIM_LOG="${RUN_ROOT}/simulator.log"
POLICY_LOG="${RUN_ROOT}/policy.log"
STOP_FILE="${RUN_ROOT}/stop_requested"

SIM_PID=""
POLICY_PID=""

cleanup() {
    touch "$STOP_FILE" 2>/dev/null || true

    [ -z "$POLICY_PID" ] || kill "$POLICY_PID" 2>/dev/null || true
    [ -z "$SIM_PID" ] || kill "$SIM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$FALCON_ENV"

export MUJOCO_GL=egl
export CYCLONEDDS_HOME="$FALCON_ENV"
export CMAKE_PREFIX_PATH="${FALCON_ENV}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${FALCON_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${FALCON_UPSTREAM}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
export RUN_ROOT
export VIDEO_PATH
export SIM2REAL

FALCON_HEAD_BEFORE="$(
    git -C "$FALCON_UPSTREAM" rev-parse HEAD
)"
FALCON_STATUS_BEFORE="$(
    git -C "$FALCON_UPSTREAM" status \
        --porcelain=v1 \
        --untracked-files=all |
    sha256sum |
    awk '{print $1}'
)"
AGILE_HEAD_BEFORE="$(
    git -C "$AGILE" rev-parse HEAD 2>/dev/null || echo UNKNOWN
)"

python "${PROJECT_ROOT}/scripts/video_simulator.py" \
    >"$SIM_LOG" 2>&1 &
SIM_PID=$!

sleep 2

if ! kill -0 "$SIM_PID" 2>/dev/null; then
    echo "VIDEO_STATUS=FAIL" > "$STATUS"
    echo "PRIMARY_REASON=VIDEO_SIMULATOR_STARTUP_FAILED" >> "$STATUS"
    cat "$SIM_LOG"
    exit 20
fi

python "${PROJECT_ROOT}/scripts/s1_01t_policy_60s.py" \
    >"$POLICY_LOG" 2>&1 &
POLICY_PID=$!

set +e
timeout --signal=TERM --kill-after=10s 190s \
    tail --pid="$POLICY_PID" -f /dev/null
POLICY_WAIT_RC=$?
set -e

if [ "$POLICY_WAIT_RC" -eq 124 ]; then
    kill "$POLICY_PID" 2>/dev/null || true
    wait "$POLICY_PID" 2>/dev/null || true
    touch "$STOP_FILE"

    echo "VIDEO_STATUS=FAIL" > "$STATUS"
    echo "PRIMARY_REASON=POLICY_TIMEOUT_DURING_VIDEO_RUN" >> "$STATUS"
    exit 21
fi

set +e
wait "$POLICY_PID"
POLICY_RC=$?
set -e
POLICY_PID=""

touch "$STOP_FILE"

set +e
timeout --signal=TERM --kill-after=10s 45s \
    tail --pid="$SIM_PID" -f /dev/null
SIM_WAIT_RC=$?
set -e

if [ "$SIM_WAIT_RC" -eq 124 ]; then
    kill "$SIM_PID" 2>/dev/null || true
fi

set +e
wait "$SIM_PID"
SIM_RC=$?
set -e
SIM_PID=""

FALCON_HEAD_AFTER="$(
    git -C "$FALCON_UPSTREAM" rev-parse HEAD
)"
FALCON_STATUS_AFTER="$(
    git -C "$FALCON_UPSTREAM" status \
        --porcelain=v1 \
        --untracked-files=all |
    sha256sum |
    awk '{print $1}'
)"
AGILE_HEAD_AFTER="$(
    git -C "$AGILE" rev-parse HEAD 2>/dev/null || echo UNKNOWN
)"

VIDEO_SIZE=0
[ ! -f "$VIDEO_PATH" ] ||
    VIDEO_SIZE="$(stat -c '%s' "$VIDEO_PATH")"

if [ "$POLICY_RC" -eq 0 ] &&
   [ "$SIM_RC" -eq 0 ] &&
   [ "$VIDEO_SIZE" -ge 100000 ] &&
   [ "$FALCON_HEAD_BEFORE" = "$FALCON_HEAD_AFTER" ] &&
   [ "$FALCON_STATUS_BEFORE" = "$FALCON_STATUS_AFTER" ]; then

    VIDEO_STATUS=PASS
    PRIMARY_REASON=EGL_OFFSCREEN_VIDEO_CREATED
    FALCON_SOURCE_MODIFIED=NO
else
    VIDEO_STATUS=FAIL
    PRIMARY_REASON=VIDEO_OR_IMMUTABILITY_GATE_FAILED
    FALCON_SOURCE_MODIFIED=YES
fi

cat > "${STATUS}.tmp" <<EOF
VIDEO_STATUS=${VIDEO_STATUS}
PRIMARY_REASON=${PRIMARY_REASON}
VIDEO_PATH=${VIDEO_PATH}
VIDEO_SIZE_BYTES=${VIDEO_SIZE}
POLICY_PROCESS_RC=${POLICY_RC}
SIMULATOR_PROCESS_RC=${SIM_RC}
MUJOCO_GL=egl
FALCON_SOURCE_MODIFIED=${FALCON_SOURCE_MODIFIED}
AGILE_ENV_MODIFIED=NO
AGILE_HEAD_BEFORE=${AGILE_HEAD_BEFORE}
AGILE_HEAD_AFTER=${AGILE_HEAD_AFTER}
FALCON_TRAINING_STARTED=NO
REAL_ROBOT_CONNECTED=NO
RUN_ROOT=${RUN_ROOT}
SIMULATOR_LOG=${SIM_LOG}
POLICY_LOG=${POLICY_LOG}
SIMULATOR_REPORT=${RUN_ROOT}/video_simulator_report.json
EOF

mv -f "${STATUS}.tmp" "$STATUS"

ln -sfn "$VIDEO_PATH" "${RUN_ROOT}/video.mp4"

cat "$STATUS"

[ "$VIDEO_STATUS" = "PASS" ]
