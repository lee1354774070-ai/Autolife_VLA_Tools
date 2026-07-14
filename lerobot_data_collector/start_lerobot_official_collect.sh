#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /home/ubuntu/lerobot_data_collector/start_lerobot_official_collect.sh [task_name] [task_text]
#   TASK_TEXT="pick the mango" bash /home/ubuntu/lerobot_data_collector/start_lerobot_official_collect.sh pick_the_mango

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="${1:-mango_pick}"
TASK_TEXT="${TASK_TEXT:-${2:-${TASK_NAME}}}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/home/ubuntu/nas14}"
BASE_DIR="${OUTPUT_BASE_DIR}/${TASK_NAME}"
DATASET_ROOT="${BASE_DIR}/dataset"
PIDFILE="${BASE_DIR}/.official_recording_pids"
CURRENT_OUTPUT="${BASE_DIR}/.current_output"
CONTROL_FIFO="${BASE_DIR}/.official_recording_control"
STATUS_FILE="${BASE_DIR}/.official_recording_status.json"
STATE_READY_FILE="${BASE_DIR}/.official_recorder_state_ready.json"
EPISODE_EVENT_FILE="${BASE_DIR}/.official_episode_event.json"
LOG_DIR="${BASE_DIR}/logs"

ROBOT_PY="${ROBOT_PY:-/home/ubuntu/miniconda3/envs/robot_env/bin/python}"
LEROBOT_PY="${LEROBOT_PY:-/home/ubuntu/miniconda3/envs/lerobot/bin/python}"
BRIDGE_SCRIPT="${SCRIPT_DIR}/shm_camera_topic_bridge.py"
RECORDER_SCRIPT="${SCRIPT_DIR}/record_lerobot_official.py"
HAND_PRODUCER_SCRIPT="${SCRIPT_DIR}/hand_camera_producer.py"
CONTROL_SCRIPT="${SCRIPT_DIR}/collector_control.py"

HOST_ROBOT_ID="$(hostname | sed -n 's/.*-\([0-9][0-9]*\)$/\1/p')"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROBOT_ID="${ROBOT_ID:-${HOST_ROBOT_ID:-283}}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
TOPIC_NODE_ID="${ROS_DOMAIN_ID}_${ROBOT_ID}"

mkdir -p "${BASE_DIR}" "${LOG_DIR}"

# Refuse to reuse a session directory while any process from its last PID file
# is still alive. Concurrent writes can corrupt LeRobot parquet/video indices.
if [ -f "${PIDFILE}" ]; then
    while read -r _ pid _; do
        if [ -n "${pid:-}" ] && kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: collection already running (PID ${pid})."
            echo "Stop it from the active collector terminal with Q or Ctrl+C."
            exit 1
        fi
    done < "${PIDFILE}"
fi

RUN_SEQ=1
# Logs are never overwritten; each restart receives the next numeric prefix.
while [ -e "${LOG_DIR}/${RUN_SEQ}.record_lerobot_official.log" ]; do
    RUN_SEQ=$((RUN_SEQ + 1))
done
LOG_PREFIX="${LOG_DIR}/${RUN_SEQ}"
CLEANUP_DONE=0

send_control() {
    python3 "${CONTROL_SCRIPT}" command --base-dir "${BASE_DIR}" "$1"
}

send_control_and_report() {
    python3 "${CONTROL_SCRIPT}" command \
        --base-dir "${BASE_DIR}" \
        --wait \
        --timeout "${CONTROL_ACK_TIMEOUT_SEC:-300}" \
        "$1"
}

cleanup_session() {
    if [ "${CLEANUP_DONE}" = "1" ]; then
        return
    fi
    CLEANUP_DONE=1

    # Give the recorder time to save pending frames and finalize metadata before
    # stopping camera processes in reverse launch order.
    if [ -n "${RECORDER_PID:-}" ] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
        send_control "quit" >/dev/null 2>&1 || kill -INT "${RECORDER_PID}" 2>/dev/null || true
        for _ in $(seq 1 120); do
            kill -0 "${RECORDER_PID}" 2>/dev/null || break
            sleep 0.25
        done
    fi

    if [ -f "${PIDFILE}" ]; then
        tac "${PIDFILE}" | while read -r role pid log; do
            if [ -z "${pid:-}" ]; then
                continue
            fi
            if kill -0 "${pid}" 2>/dev/null; then
                echo "Stopping ${role} ${pid} ..."
                kill -INT "${pid}" 2>/dev/null || true
                for _ in $(seq 1 20); do
                    kill -0 "${pid}" 2>/dev/null || break
                    sleep 0.2
                done
            fi
            if kill -0 "${pid}" 2>/dev/null; then
                echo "Terminating ${role} ${pid} ..."
                kill -TERM "${pid}" 2>/dev/null || true
            fi
        done
    fi

    rm -f "${PIDFILE}" "${CONTROL_FIFO}" "${STATUS_FILE}" "${STATE_READY_FILE}" "${EPISODE_EVENT_FILE}"
    echo ""
    python3 "${CONTROL_SCRIPT}" summary --dataset-root "${DATASET_ROOT}"
    python3 "${CONTROL_SCRIPT}" sync-report --sync-log "${DATASET_ROOT}/sync_log.jsonl"
    echo "=============================================="
}

on_interrupt() {
    echo ""
    echo "Interrupted. Stopping collection ..."
    exit 0
}

trap cleanup_session EXIT
trap on_interrupt INT TERM

echo "=============================================="
echo "  Official LeRobotDataset batch collection"
echo "  task name   : ${TASK_NAME}"
echo "  task text   : ${TASK_TEXT}"
echo "  dataset root: ${DATASET_ROOT}"
echo "  topic id    : ${TOPIC_NODE_ID}"
echo "  run log seq : ${RUN_SEQ}"
echo "=============================================="

set +u
source /opt/ros/jazzy/setup.bash
set -u
# Preserve paths exported by ROS, then prepend each conda environment's native
# libraries only for the process that needs them.
ROS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ROS_PYTHONPATH="${PYTHONPATH:-}"
ROBOT_ENV_LIB="/home/ubuntu/miniconda3/envs/robot_env/lib"
LEROBOT_ENV_LIB="/home/ubuntu/miniconda3/envs/lerobot/lib"
COLLECT_CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>127.0.0.1</NetworkInterfaceAddress></General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>120</MaxAutoParticipantIndex></Discovery></Domain></CycloneDDS>'
CAMERA_ONLY="${CAMERA_ONLY:-0}"
COLLECT_FPS="${COLLECT_FPS:-30}"
WITH_HEAD="${WITH_HEAD:-0}"
WITH_WAIST="${WITH_WAIST:-0}"
WITH_DEPTH="${WITH_DEPTH:-0}"
BRIDGE_CAMERAS=(rgbd_head_color hand_left hand_right)
IMAGE_SOURCE="${IMAGE_SOURCE:-shm}"
IMAGE_POLL_FPS="${IMAGE_POLL_FPS:-120}"
RECORDER_CAMERA_ARGS=()
RECORDER_FEATURE_ARGS=()

# Feature switches are intentionally session-level settings.  LeRobot fixes
# feature shapes when a dataset root is created, so changing one of these while
# resuming an existing root will produce a clear recorder startup error.
if [ "${WITH_HEAD}" = "1" ]; then
    RECORDER_FEATURE_ARGS+=(--with-head)
fi
if [ "${WITH_WAIST}" = "1" ]; then
    RECORDER_FEATURE_ARGS+=(--with-waist)
fi
if [ -z "${START_HAND_PRODUCER+x}" ] && { fuser /dev/video12 >/dev/null 2>&1 || fuser /dev/video14 >/dev/null 2>&1; }; then
    START_HAND_PRODUCER=0
    echo "  mode        : hand camera devices busy; reusing existing SHM streams"
fi
if [ "${CAMERA_ONLY}" = "1" ]; then
    START_HAND_PRODUCER=0
    FALLBACK_ACTION_TO_STATE=1
    BRIDGE_CAMERAS=(rgbd_head_color)
    RECORDER_CAMERA_ARGS=(--camera rgbd_head_color)
    echo "  mode        : CAMERA_ONLY=1 (rgbd_head_color only, action=state fallback)"
fi
if [ "${WITH_DEPTH}" = "1" ]; then
    BRIDGE_CAMERAS+=(rgbd_head_depth)
    RECORDER_FEATURE_ARGS+=(--with-depth)
fi

echo "  joint data  : head=${WITH_HEAD}, waist=${WITH_WAIST}"
echo "  depth data  : ${WITH_DEPTH}"
echo "  image FIFO  : ${SYNC_IMAGE_BUFFER_SIZE:-16} frames/camera"

rm -f "${CONTROL_FIFO}" "${STATUS_FILE}" "${STATE_READY_FILE}" "${EPISODE_EVENT_FILE}"
mkfifo "${CONTROL_FIFO}"
: > "${PIDFILE}"

RECORDER_ARGS=(
    "${RECORDER_SCRIPT}"
    --output-dir "${DATASET_ROOT}"
    --repo-id "local/${TASK_NAME}"
    --task-name "${TASK_TEXT}"
    --fps "${COLLECT_FPS}"
    --image-source "${IMAGE_SOURCE}"
    --image-poll-fps "${IMAGE_POLL_FPS}"
    --state-warmup-sec "${STATE_READY_TIMEOUT_SEC:-10}"
    --camera-warmup-sec "${CAMERA_WARMUP_SEC:-10}"
    --state-ready-file "${STATE_READY_FILE}"
    --episode-event-file "${EPISODE_EVENT_FILE}"
    --min-cameras "${MIN_CAMERAS:-1}"
    --max-sync-delta-sec "${MAX_SYNC_DELTA_SEC:-0.03}"
    --max-image-age-sec "${MAX_IMAGE_AGE_SEC:-0.15}"
    --max-state-age-sec "${MAX_STATE_AGE_SEC:-0.15}"
    --max-state-interpolation-gap-sec "${MAX_STATE_INTERPOLATION_GAP_SEC:-0.05}"
    --max-action-hold-sec "${MAX_ACTION_HOLD_SEC:-0.5}"
    --max-frame-interval-error-ratio "${MAX_FRAME_INTERVAL_ERROR_RATIO:-0.45}"
    --sync-image-buffer-size "${SYNC_IMAGE_BUFFER_SIZE:-16}"
    --sync-signal-buffer-size "${SYNC_SIGNAL_BUFFER_SIZE:-64}"
    --action-mode "${ACTION_MODE:-status_target}"
    --vcodec "${COLLECT_VCODEC:-h264}"
    --depth-vcodec "${DEPTH_VCODEC:-hevc}"
    --depth-min "${DEPTH_MIN:-0.05}"
    --depth-max "${DEPTH_MAX:-10.0}"
    --depth-shift "${DEPTH_SHIFT:-3.5}"
    --batch-encoding-size "${BATCH_ENCODING_SIZE:-1}"
    --encoder-queue-maxsize "${ENCODER_QUEUE_MAXSIZE:-30}"
    --state-topic "/topic_arm_whole_body_and_gripper_current_joints_status_${TOPIC_NODE_ID}"
    --action-arm-topic "/topic_arm_whole_body_target_joints_position_${TOPIC_NODE_ID}"
    --action-gripper-topic "/topic_arm_gripper_target_joints_position_${TOPIC_NODE_ID}"
    --action-eef-topic "/topic_arm_target_robot_eef_pose_${TOPIC_NODE_ID}"
    --action-height-topic "/topic_arm_target_robot_height_z_${TOPIC_NODE_ID}"
    --control-fifo "${CONTROL_FIFO}"
    --status-file "${STATUS_FILE}"
    "${RECORDER_CAMERA_ARGS[@]}"
    "${RECORDER_FEATURE_ARGS[@]}"
)
if [ -n "${SYNC_REFERENCE_CAMERA:-}" ]; then
    RECORDER_ARGS+=(--sync-reference-camera "${SYNC_REFERENCE_CAMERA}")
fi
if [ "${DEPTH_USE_LOG:-1}" = "0" ]; then
    RECORDER_ARGS+=(--no-depth-use-log)
fi
if [ "${FALLBACK_ACTION_TO_STATE:-0}" = "1" ]; then
    RECORDER_ARGS+=(--fallback-action-to-state)
fi
if [ -n "${VIDEO_CRF:-}" ]; then
    RECORDER_ARGS+=(--video-crf "${VIDEO_CRF}")
fi
if [ -n "${VIDEO_PRESET:-}" ]; then
    RECORDER_ARGS+=(--video-preset "${VIDEO_PRESET}")
fi
if [ -n "${VIDEO_GOP:-}" ]; then
    RECORDER_ARGS+=(--video-gop "${VIDEO_GOP}")
fi
if [ -n "${ENCODER_THREADS:-}" ]; then
    RECORDER_ARGS+=(--encoder-threads "${ENCODER_THREADS}")
fi
if [ -n "${VIDEO_FILES_SIZE_IN_MB:-}" ]; then
    RECORDER_ARGS+=(--video-files-size-in-mb "${VIDEO_FILES_SIZE_IN_MB}")
fi
if [ -n "${DATA_FILES_SIZE_IN_MB:-}" ]; then
    RECORDER_ARGS+=(--data-files-size-in-mb "${DATA_FILES_SIZE_IN_MB}")
fi
if [ "${STREAMING_ENCODING:-0}" = "1" ]; then
    RECORDER_ARGS+=(--streaming-encoding)
fi

nohup env CYCLONEDDS_URI="${COLLECT_CYCLONEDDS_URI}" LD_LIBRARY_PATH="${LEROBOT_ENV_LIB}:${ROS_LD_LIBRARY_PATH}" PYTHONPATH="${ROS_PYTHONPATH}" "${LEROBOT_PY}" "${RECORDER_ARGS[@]}" > "${LOG_PREFIX}.record_lerobot_official.log" 2>&1 &
RECORDER_PID=$!
echo "recorder ${RECORDER_PID} ${LOG_PREFIX}.record_lerobot_official.log" >> "${PIDFILE}"
echo "${DATASET_ROOT}" > "${CURRENT_OUTPUT}"

echo "  recorder PID      : ${RECORDER_PID}"
echo "Waiting for the first complete joint-state packet ..."
python3 "${CONTROL_SCRIPT}" wait-ready \
    --path "${STATE_READY_FILE}" \
    --pid "${RECORDER_PID}" \
    --timeout "${STATE_READY_TIMEOUT_SEC:-10}"

# Camera production starts only after the recorder is actively buffering joint
# state. Video frames can then act as anchors for nearest-state lookup without
# losing the first samples to process startup ordering.
if [ "${START_HAND_PRODUCER:-1}" = "1" ]; then
    nohup env CYCLONEDDS_URI="${COLLECT_CYCLONEDDS_URI}" LD_LIBRARY_PATH="${ROBOT_ENV_LIB}:${ROS_LD_LIBRARY_PATH}" "${ROBOT_PY}" "${HAND_PRODUCER_SCRIPT}" > "${LOG_PREFIX}.hand_camera_producer.log" 2>&1 &
    HAND_PID=$!
    echo "hand_producer ${HAND_PID} ${LOG_PREFIX}.hand_camera_producer.log" >> "${PIDFILE}"
    echo "  hand producer PID : ${HAND_PID}"
else
    echo "  hand producer     : skipped (START_HAND_PRODUCER=0)"
fi

if [ "${IMAGE_SOURCE}" = "ros" ]; then
    nohup env CYCLONEDDS_URI="${COLLECT_CYCLONEDDS_URI}" LD_LIBRARY_PATH="${ROBOT_ENV_LIB}:${ROS_LD_LIBRARY_PATH}" PYTHONPATH="${ROS_PYTHONPATH}" "${ROBOT_PY}" "${BRIDGE_SCRIPT}" --cameras "${BRIDGE_CAMERAS[@]}" --fps "${COLLECT_FPS}" > "${LOG_PREFIX}.camera_bridge.log" 2>&1 &
    BRIDGE_PID=$!
    echo "camera_bridge ${BRIDGE_PID} ${LOG_PREFIX}.camera_bridge.log" >> "${PIDFILE}"
    echo "  camera source     : ROS topics via SHM bridge"
    echo "  camera bridge PID : ${BRIDGE_PID}"
else
    echo "  camera source     : direct SHM (poll=${IMAGE_POLL_FPS} Hz)"
fi

echo "  logs              : ${LOG_PREFIX}.*.log"
echo ""
echo "Waiting for camera warmup and dataset initialization ..."
python3 "${CONTROL_SCRIPT}" cameras \
    --dataset-root "${DATASET_ROOT}" \
    --recorder-log "${LOG_PREFIX}.record_lerobot_official.log" \
    --pid "${RECORDER_PID}" \
    --timeout "${CAMERA_DETECTION_TIMEOUT_SEC:-15}"

echo "============================================================"
echo "Interactive controls"
echo "  Enter   - Start a new episode"
echo "  S / s   - Save the current episode and pause"
echo "  D / d   - Discard the current episode and pause"
echo "  Q / q   - Save pending data and quit"
echo "============================================================"

while true; do
    if [ -f "${EPISODE_EVENT_FILE}" ]; then
        echo ""
        python3 "${CONTROL_SCRIPT}" episode-event --path "${EPISODE_EVENT_FILE}" || true
    fi

    if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
        echo ""
        echo "Recorder process exited. Check ${LOG_PREFIX}.record_lerobot_official.log"
        break
    fi

    if IFS= read -rsn1 -t 0.2 key; then
        case "${key}" in
            "")
                echo ""
                echo "[control] start"
                send_control "start" || true
                ;;
            [sS])
                echo ""
                echo "[control] save"
                send_control_and_report "save" || true
                ;;
            [dD])
                echo ""
                echo "[control] discard"
                send_control_and_report "discard" || true
                ;;
            [qQ])
                echo ""
                echo "[control] quit"
                send_control "quit" || true
                break
                ;;
            *)
                ;;
        esac
    fi
done
