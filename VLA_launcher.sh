#!/usr/bin/env bash

# ============================================================
# CloudButterfly Continuous Fake VLA Runtime
#
# 说明：
#   这是一个终端演示脚本。
#   不会真实加载模型、连接相机或控制机器人。
#
# 运行：
#   chmod +x ~/VLA_launcher.sh
#   ~/VLA_launcher.sh
#
# 自定义任务：
#   ~/VLA_launcher.sh "Pick up the red box and place it on the shelf"
#
# 停止：
#   Ctrl+C
# ============================================================

set -u

MODEL_NAME="CloudButterfly-VLA-7B"
ROBOT_NAME="CloudButterfly Dual-Arm Humanoid"
DEVICE="cuda:0"

TASK="${1:-Pick up the red box and place it on the shelf}"

VERSION="0.9.7-dev"

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
GRAY="\033[0;90m"
BOLD="\033[1m"
RESET="\033[0m"

frame=0
cycle=0
latency_sum=0
latency_count=0


cleanup() {
    echo
    echo
    echo -e "${YELLOW}[VLA] Ctrl+C received. Stopping continuous runtime...${RESET}"
    sleep 0.3

    echo -e "${GREEN}[ROBOT] Active action cancelled.${RESET}"
    echo -e "${GREEN}[ROBOT] Motion command stream stopped.${RESET}"
    echo -e "${GREEN}[ROBOT] Robot entered safe idle state.${RESET}"

    echo
    echo -e "${CYAN}[SESSION] Completed cycles: ${cycle}${RESET}"
    echo -e "${CYAN}[SESSION] Total control frames: ${frame}${RESET}"

    if (( latency_count > 0 )); then
        average_latency=$(awk \
            -v sum="$latency_sum" \
            -v count="$latency_count" \
            'BEGIN {printf "%.1f", sum / count}')

        echo -e "${CYAN}[SESSION] Average policy latency: ${average_latency} ms${RESET}"
    fi

    echo -e "${GREEN}[VLA] Runtime shutdown complete.${RESET}"
    exit 0
}

trap cleanup INT TERM


type_text() {
    local text="$1"
    local delay="${2:-0.012}"

    for ((i = 0; i < ${#text}; i++)); do
        printf "%s" "${text:$i:1}"
        sleep "$delay"
    done

    printf "\n"
}


progress_bar() {
    local label="$1"
    local total="${2:-28}"

    printf "${CYAN}%-30s${RESET} [" "$label"

    for ((i = 0; i < total; i++)); do
        printf "█"
        sleep 0.025
    done

    printf "] ${GREEN}done${RESET}\n"
}


print_logo() {
    echo -e "${BOLD}${BLUE}"

    cat <<'EOF'
   ____ _                 _
  / ___| | ___  _   _  __| |
 | |   | |/ _ \| | | |/ _` |
 | |___| | (_) | |_| | (_| |
  \____|_|\___/ \__,_|\__,_|

  ____        _   _             __ _
 | __ ) _   _| |_| |_ ___ _ __/ _| |_   _
 |  _ \| | | | __| __/ _ \ '__| |_| | | |
 | |_) | |_| | |_| ||  __/ |  |  _| |_| |
 |____/ \__,_|\__|\__\___|_|  |_|  \__, |
                                    |___/

                 CloudButterfly
EOF

    echo -e "${RESET}"
}


print_header() {
    clear

    print_logo

    echo -e "${GRAY}CloudButterfly VLA Runtime ${VERSION}${RESET}"
    echo -e "${GRAY}Build: CUDA 12.4 | PyTorch 2.7 | TensorRT enabled${RESET}"
    echo
}


initialize_runtime() {
    type_text "[SYSTEM] Initializing Vision-Language-Action runtime..." 0.015
    sleep 0.3

    progress_bar "Checking CUDA device"

    echo -e "${GREEN}[CUDA] NVIDIA H200 detected${RESET}"
    echo -e "${GREEN}[CUDA] Available VRAM: 139.6 GB${RESET}"
    echo -e "${GREEN}[CUDA] Compute capability: 9.0${RESET}"
    echo

    progress_bar "Loading vision encoder"
    echo -e "${GREEN}[MODEL] SigLIP vision encoder loaded${RESET}"

    progress_bar "Loading language backbone"
    echo -e "${GREEN}[MODEL] Language backbone loaded${RESET}"

    progress_bar "Loading action expert"
    echo -e "${GREEN}[MODEL] Flow-matching action head loaded${RESET}"

    progress_bar "Allocating KV cache"
    echo -e "${GREEN}[MODEL] Runtime cache allocated${RESET}"

    progress_bar "Compiling control graph"
    echo -e "${GREEN}[MODEL] Control graph optimized${RESET}"

    echo
    echo -e "${CYAN}[MODEL] Name:       ${MODEL_NAME}${RESET}"
    echo -e "${CYAN}[MODEL] Device:     ${DEVICE}${RESET}"
    echo -e "${CYAN}[MODEL] Precision:  bfloat16${RESET}"
    echo -e "${CYAN}[MODEL] Parameters: 7.3B${RESET}"
    echo -e "${CYAN}[MODEL] Action dim: 18${RESET}"
    echo -e "${CYAN}[MODEL] Horizon:    50 steps${RESET}"
    echo
}


connect_robot() {
    type_text "[ROBOT] Connecting to robot controller..." 0.015
    sleep 0.5

    echo -e "${GREEN}[ROBOT] Connected: ${ROBOT_NAME}${RESET}"
    echo -e "${GREEN}[ROBOT] Control frequency: 50 Hz${RESET}"
    echo -e "${GREEN}[ROBOT] Joint state stream synchronized${RESET}"
    echo -e "${GREEN}[ROBOT] Safety controller active${RESET}"
    echo
}


start_sensors() {
    type_text "[SENSORS] Starting camera streams..." 0.015
    sleep 0.3

    echo -e "${GREEN}[CAMERA] head_rgb     1280x720 @ 30 FPS${RESET}"
    echo -e "${GREEN}[CAMERA] left_wrist   640x480  @ 30 FPS${RESET}"
    echo -e "${GREEN}[CAMERA] right_wrist  640x480  @ 30 FPS${RESET}"
    echo -e "${GREEN}[SENSOR] joint_state  18 DoF   @ 50 Hz${RESET}"
    echo -e "${GREEN}[SENSORS] All observation streams ready${RESET}"
    echo
}


analyze_task() {
    echo -e "${BOLD}${YELLOW}[USER INSTRUCTION]${RESET} ${TASK}"
    echo

    type_text "[VLA] Encoding multimodal observation..." 0.012
    sleep 0.4

    echo -e "${GREEN}[VISION] Detected objects:${RESET}"
    echo "         - red box      confidence=0.982  position=[0.61, -0.18, 0.82]"
    echo "         - shelf        confidence=0.996  position=[1.14,  0.06, 1.02]"
    echo "         - table        confidence=0.991  position=[0.42, -0.03, 0.74]"
    echo

    type_text "[VLA] Interpreting language instruction..." 0.012
    sleep 0.4

    echo -e "${GREEN}[PLANNER] Generated semantic plan:${RESET}"
    echo "          1. Locate target object"
    echo "          2. Estimate dual-arm grasp poses"
    echo "          3. Move both end effectors to pre-grasp poses"
    echo "          4. Close grippers and verify grasp"
    echo "          5. Lift and transport object"
    echo "          6. Place object at target location"
    echo "          7. Release object and return to idle"
    echo

    echo -e "${GREEN}[POLICY] Action chunk generated: shape=[50, 18]${RESET}"
    echo -e "${GREEN}[POLICY] Closed-loop replanning enabled${RESET}"
    echo
}


calculate_random_values() {
    latency=$((18 + RANDOM % 11))

    confidence=$(awk \
        -v r="$RANDOM" \
        'BEGIN {printf "%.3f", 0.925 + (r % 70) / 1000}')

    action_norm=$(awk \
        -v r="$RANDOM" \
        'BEGIN {printf "%.3f", 0.080 + (r % 450) / 1000}')
}


calculate_grippers() {
    local current_stage="$1"
    local current_step="$2"

    if [[ "$current_stage" == "CLOSING_GRIPPERS" ||
          "$current_stage" == "LIFTING_OBJECT" ||
          "$current_stage" == "MOVING_TO_SHELF" ]]; then

        gripper_left=$(awk \
            -v r="$RANDOM" \
            'BEGIN {printf "%.2f", 0.90 + (r % 7) / 100}')

        gripper_right=$(awk \
            -v r="$RANDOM" \
            'BEGIN {printf "%.2f", 0.90 + (r % 7) / 100}')

    elif [[ "$current_stage" == "PLACING_OBJECT" ]]; then

        gripper_left=$(awk \
            -v s="$current_step" \
            'BEGIN {
                value = 0.95 - s * 0.083;
                if (value < 0.12) {
                    value = 0.12;
                }
                printf "%.2f", value
            }')

        gripper_right=$(awk \
            -v s="$current_step" \
            'BEGIN {
                value = 0.95 - s * 0.084;
                if (value < 0.11) {
                    value = 0.11;
                }
                printf "%.2f", value
            }')

    else
        gripper_left=$(awk \
            -v r="$RANDOM" \
            'BEGIN {printf "%.2f", 0.10 + (r % 9) / 100}')

        gripper_right=$(awk \
            -v r="$RANDOM" \
            'BEGIN {printf "%.2f", 0.10 + (r % 9) / 100}')
    fi
}


print_stage_result() {
    local current_stage="$1"

    case "$current_stage" in
        "CLOSING_GRIPPERS")
            echo -e "${GREEN}[GRASP] Contact detected on both grippers${RESET}"
            echo -e "${GREEN}[GRASP] Grasp stability score: 0.963${RESET}"
            ;;

        "LIFTING_OBJECT")
            echo -e "${GREEN}[VISION] Object motion verified: delta_z=+0.201m${RESET}"
            ;;

        "MOVING_TO_SHELF")
            echo -e "${GREEN}[PLANNER] Target shelf region reached${RESET}"
            ;;

        "PLACING_OBJECT")
            echo -e "${GREEN}[VISION] Placement region verified${RESET}"
            echo -e "${GREEN}[GRASP] Object released successfully${RESET}"
            ;;
    esac
}


print_cycle_metrics() {
    local average_latency
    local control_rate

    average_latency=$(awk \
        -v sum="$latency_sum" \
        -v count="$latency_count" \
        'BEGIN {
            if (count == 0) {
                printf "0.0"
            } else {
                printf "%.1f", sum / count
            }
        }')

    control_rate=$(awk \
        -v latency="$average_latency" \
        'BEGIN {
            if (latency == 0) {
                printf "0.0"
            } else {
                printf "%.1f", 1000 / latency
            }
        }')

    echo -e "${BOLD}${GREEN}[SUCCESS] Control cycle ${cycle} completed.${RESET}"
    echo

    echo -e "${CYAN}[METRICS] Completed cycles: ${cycle}${RESET}"
    echo -e "${CYAN}[METRICS] Total control steps: ${frame}${RESET}"
    echo -e "${CYAN}[METRICS] Average policy latency: ${average_latency} ms${RESET}"
    echo -e "${CYAN}[METRICS] Effective control rate: ${control_rate} Hz${RESET}"
    echo -e "${CYAN}[METRICS] Final task confidence: 0.972${RESET}"
    echo -e "${CYAN}[METRICS] Grasp verification: passed${RESET}"
    echo -e "${CYAN}[METRICS] Safety violations: 0${RESET}"
    echo
}


STAGES=(
    "PERCEIVING_SCENE"
    "PLANNING_GRASP"
    "APPROACHING_OBJECT"
    "ALIGNING_GRIPPERS"
    "CLOSING_GRIPPERS"
    "LIFTING_OBJECT"
    "MOVING_TO_SHELF"
    "PLACING_OBJECT"
)


print_header
initialize_runtime
connect_robot
start_sensors
analyze_task

echo -e "${BOLD}${CYAN}Starting continuous closed-loop VLA control...${RESET}"
echo -e "${GRAY}The runtime will continue until Ctrl+C is pressed.${RESET}"
echo


while true; do
    cycle=$((cycle + 1))

    echo
    echo -e "${BOLD}${CYAN}============================================================${RESET}"
    echo -e "${BOLD}${CYAN}[VLA] Starting control cycle ${cycle}${RESET}"
    echo -e "${BOLD}${CYAN}============================================================${RESET}"
    echo

    echo -e "${GRAY}[OBSERVATION] Capturing new multimodal observation...${RESET}"
    sleep 0.4

    echo -e "${GREEN}[OBSERVATION] Head camera frame synchronized${RESET}"
    echo -e "${GREEN}[OBSERVATION] Wrist camera frames synchronized${RESET}"
    echo -e "${GREEN}[OBSERVATION] Joint states synchronized${RESET}"
    echo

    for stage_index in "${!STAGES[@]}"; do
        stage="${STAGES[$stage_index]}"

        echo -e "${BOLD}${YELLOW}[STATE] ${stage}${RESET}"

        for step in $(seq 1 10); do
            frame=$((frame + 1))

            calculate_random_values
            calculate_grippers "$stage" "$step"

            latency_sum=$((latency_sum + latency))
            latency_count=$((latency_count + 1))

            printf "\r${GRAY}"
            printf "[cycle=%03d frame=%06d] " "$cycle" "$frame"
            printf "obs=ready  "
            printf "policy_conf=%s  " "$confidence"
            printf "action_norm=%s  " "$action_norm"
            printf "grip=[%s,%s]  " "$gripper_left" "$gripper_right"
            printf "latency=%dms" "$latency"
            printf "${RESET}"

            sleep 0.10
        done

        printf "\n"

        print_stage_result "$stage"

        echo -e "${GREEN}[STATE] ${stage} completed${RESET}"
        echo

        sleep 0.25
    done

    print_cycle_metrics

    echo -e "${GREEN}[ROBOT] Returning to observation pose...${RESET}"
    sleep 0.5

    echo -e "${GREEN}[VLA] Waiting for next observation...${RESET}"
    sleep 1
done