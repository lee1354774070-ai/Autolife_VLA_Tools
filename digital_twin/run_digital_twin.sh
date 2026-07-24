#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/home/wayne-cb/isaacsim_0_6_1}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"

if [[ ! -x "${ISAAC_SIM_PATH}/python.sh" ]]; then
    echo "Isaac Sim python.sh not found: ${ISAAC_SIM_PATH}/python.sh" >&2
    echo "Set ISAAC_SIM_PATH to the Isaac Sim installation directory." >&2
    exit 1
fi
if [[ ! -f "${ROS_SETUP}" ]]; then
    echo "ROS setup file not found: ${ROS_SETUP}" >&2
    exit 1
fi

# ROS setup scripts reference optional variables that are incompatible with
# nounset while they are being sourced.
set +u
# shellcheck disable=SC1090
source "${ROS_SETUP}"
set -u
export PYTHONUNBUFFERED=1

# Keep image transport isolated from the latency-sensitive joint mirror.  The
# three-camera viewer can be enabled explicitly after pose synchronization has
# been confirmed.
show_cameras="${SHOW_CAMERAS:-0}"
robot_id=""
robot_host=""
robot_user="ubuntu"
mirror_args=()
while (($#)); do
    case "$1" in
        --camera-viewer|--show-camera-viewer)
            show_cameras=1
            shift
            ;;
        --no-camera-viewer)
            show_cameras=0
            shift
            ;;
        --robot-id)
            mirror_args+=("$1" "$2")
            robot_id="$2"
            shift 2
            ;;
        --robot-id=*)
            mirror_args+=("$1")
            robot_id="${1#*=}"
            shift
            ;;
        --robot-host)
            mirror_args+=("$1" "$2")
            robot_host="$2"
            shift 2
            ;;
        --robot-host=*)
            mirror_args+=("$1")
            robot_host="${1#*=}"
            shift
            ;;
        --robot-user)
            mirror_args+=("$1" "$2")
            robot_user="$2"
            shift 2
            ;;
        --robot-user=*)
            mirror_args+=("$1")
            robot_user="${1#*=}"
            shift
            ;;
        *)
            mirror_args+=("$1")
            shift
            ;;
    esac
done

if [[ "${show_cameras}" == "1" && -n "${robot_id}" ]]; then
    camera_args=(--robot-id "${robot_id}" --robot-user "${robot_user}" --parent-pid "$$")
    if [[ -n "${robot_host}" ]]; then
        camera_args+=(--robot-host "${robot_host}")
    fi
    if [[ -n "${DISPLAY:-}" ]] && command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal \
            --title="AutoLife ${robot_id} RGB Cameras" \
            -- "${SCRIPT_DIR}/run_camera_viewer.sh" "${camera_args[@]}" &
        echo "[digital_twin] opening three-camera viewer in another Ubuntu window."
    else
        echo "[digital_twin] camera viewer was not opened: DISPLAY or gnome-terminal is unavailable." >&2
        echo "[digital_twin] run manually: ${SCRIPT_DIR}/run_camera_viewer.sh --robot-id ${robot_id}" >&2
    fi
fi

exec "${ISAAC_SIM_PATH}/python.sh" "${SCRIPT_DIR}/mirror_robot.py" "${mirror_args[@]}"
