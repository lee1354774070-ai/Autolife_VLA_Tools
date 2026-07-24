# Robot Model Deployment

Deployment code is grouped by responsibility:

```text
deploy/
├── pi05/          # PI0.5 and quantized PI0.5 entry points
├── lingbot_va/    # LingBot-VA local/remote inference and Thor runtime
├── common/        # Shared session and transport helpers
└── tests/         # Tests grouped by component
```

## Local PI0.5 deployment

Run a LeRobot PI0.5 checkpoint on an Autolife robot using the collector's
camera names, SHM input, and canonical joint schema.

## Start

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export ROS_DOMAIN_ID=0
export ROBOT_ID=283
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>127.0.0.1</NetworkInterfaceAddress></General></Domain></CycloneDDS>'
export MODEL_DIR=/home/ubuntu/your_model/pretrained_model

python /home/ubuntu/deploy/pi05/deploy_pi05.py \
  --task "pick up the water bottle"
```

`--model-dir` overrides `MODEL_DIR`. The repository default remains the legacy
`004500/pretrained_model` path for compatibility.

## Persistent interactive session

Use `--interactive` when the same model should remain in GPU memory across
multiple tasks. The process creates ROS2 and SHM readers immediately, but it
does not load the PI0.5 weights until `enable` is entered.

```bash
python /home/ubuntu/deploy/pi05/deploy_pi05.py \
  --interactive \
  --model-dir /mnt/nas14/pi05_models/MZJ/pi05_baseline_003000_pretrained_model/pretrained_model \
  --tokenizer-dir /home/ubuntu/pi05_assets/paligemma-3b-pt-224
```

Enter one command per line in the same terminal:

```text
enable
start pick up the water bottle
stop
continue
exit
```

- `enable`: loads the model, waits for fresh joint/camera data, and performs a
  no-publish warm-up. The model and CUDA compile cache stay resident.
- `start <task text>`: resets PI0.5 action-chunk state and begins a new task.
- `stop`: pauses publishing but retains the loaded model and current task.
- `continue`: keeps the task and loaded model, but resets the old action chunk
  and resumes from a fresh robot observation.
- `exit`: stops publishing, releases ROS2/SHM/GPU resources, and ends the
  process.

`status` prints the current session state and `help` lists the available
commands. Only `start` and `continue` can publish robot commands. Use
`--dry-run` to exercise the same session without publishing.

The default visual input is:

```text
observation.images.rgbd_head_color
observation.images.hand_left
observation.images.hand_right
```

The checkpoint must contain exactly the selected LeRobot image feature keys and
the matching state/action dimensions. The deployer validates this before it
publishes a command.

## Action chunks and RTC

Without RTC, PI0.5 runs synchronously and executes `n_action_steps` actions
from each predicted chunk. Omit the option to keep the value saved in the
checkpoint, or override it at startup:

```bash
python pi05/deploy_pi05.py --interactive --n-action-steps 10
```

At `--hz 10`, `--n-action-steps 10` requests a new chunk after about one
second of executed actions. Inference itself blocks the non-RTC control loop.

Enable Real-Time Chunking when inference should run in a background thread
while the control loop continues consuming the previous chunk:

```bash
python pi05/deploy_pi05.py \
  --interactive \
  --rtc \
  --rtc-refresh-steps 10 \
  --rtc-execution-horizon 10
```

- `--rtc`: switches PI0.5 from `select_action()` to RTC-compatible
  `predict_action_chunk()` and enables asynchronous chunk merging.
- `--rtc-refresh-steps`: starts a new VLM inference after this many actions
  have been consumed from the latest chunk. The approximate requested refresh
  interval is `rtc_refresh_steps / hz` seconds.
- `--rtc-execution-horizon`: number of leftover actions supplied to PI0.5 as
  the correction prefix; this is not the VLM refresh interval.
- `--rtc-max-guidance-weight`: controls the maximum RTC prefix-guidance weight.

When `--rtc-refresh-steps` is omitted, the script derives its interval from
LeRobot's default of refreshing with 30 actions remaining. For the common
`chunk_size=50` model, that starts an inference after 20 actions, or about 2
seconds at 10 Hz. The
actual refresh rate cannot exceed the model's inference throughput. Choose a
smaller value early enough that the remaining actions cover one inference;
otherwise the action queue can become empty and command publication waits for
the new chunk.

`--n-action-steps` controls only non-RTC execution. In RTC mode,
`--rtc-refresh-steps` controls refresh timing and `--rtc-execution-horizon`
controls the overlap used to correct the next chunk.

## Model variants

```bash
# Old 16-D, one-camera checkpoint.
python pi05/deploy_pi05.py --model-dir /home/ubuntu/004500/pretrained_model \
  --camera rgbd_head_color --task "pick up the water bottle"

# New default: 16-D joints and three RGB cameras.
python pi05/deploy_pi05.py --model-dir /home/ubuntu/new_model/pretrained_model

# 19-D: append head joints.
python pi05/deploy_pi05.py --with-head

# 20-D: append waist joints.
python pi05/deploy_pi05.py --with-waist

# 23-D: head and waist.
python pi05/deploy_pi05.py --with-head --with-waist

# Add the collector-compatible head depth feature.
python pi05/deploy_pi05.py --with-depth
```

The selected switches must match training. Disabled head or waist groups remain
at their latest measured positions. `--with-depth` adds
`observation.images.rgbd_head_depth` and applies the same depth quantization as
the collector.

## Cameras and safety

Use `--camera` repeatedly to select RGB inputs. Default camera names are
`rgbd_head_color`, `hand_left`, and `hand_right`; depth names use
`--depth-camera`. Camera timestamps are checked against
`--sync-reference-camera` and `--max-image-delta-sec`.

Always validate a checkpoint before moving the robot:

```bash
python pi05/deploy_pi05.py --dry-run --max-steps 30
python pi05/deploy_pi05.py --help
python pi05/deploy_pi05.py --with-depth --help
```

The deployer reads images directly from `/dev/shm`, converts BGR to model RGB,
and publishes the same whole-body and gripper topics used by the collector.

## Quantized PI0.5

Install the TorchAO version recorded in the quantized checkpoint manifest, then
use the dedicated entry point:

```bash
python /home/ubuntu/Autolife_VLA_Tools/deploy/pi05/deploy_pi05_light_weight.py \
  --interactive \
  --model-dir /path/to/pretrained_model_int8wo \
  --tokenizer-dir /home/ubuntu/pi05_assets/paligemma-3b-pt-224 \
  --dry-run
```

It supports the same cameras, joint options, safety checks, and interactive
commands as `pi05/deploy_pi05.py`. Its default compile mode is
`max-autotune-no-cudagraphs`; override it with `--compile-mode disabled` when
debugging eager PyTorch.

The loader rejects incompatible PyTorch, TorchAO, or LeRobot versions before
initializing robot control. Re-quantize the model in the deployment environment
when the recorded and installed versions differ.

## LingBot-VA LoRA

`lingbot_va/deploy_lingbot_va.py` reuses the same ROS2, direct-SHM camera, canonical joint,
and interactive safety controls. It maps the physical cameras in this order:

```text
rgbd_head_color -> observation.images.cam_high
hand_left       -> observation.images.cam_left_wrist
hand_right      -> observation.images.cam_right_wrist
```

The current checkpoint controls only the 16-D base schema (two 7-D arms and
two grippers). Depth, head, and waist inputs are intentionally excluded.

Activate the robot environment and start with a non-publishing dry-run:

```bash
source /opt/ros/jazzy/setup.bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export ROS_DOMAIN_ID=0
export ROBOT_ID=283
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><NetworkInterfaceAddress>127.0.0.1</NetworkInterfaceAddress></General></Domain></CycloneDDS>'

python /home/ubuntu/deploy/lingbot_va/deploy_lingbot_va.py \
  --interactive \
  --model-dir /home/ubuntu/models/LingBot-VA \
  --base-model /path/to/lerobot_lingbot_va_base \
  --wan-model /path/to/lingbot_wan_frozen_modules \
  --offline
```

The Wan directory only needs `vae/`, `text_encoder/`, and `tokenizer/`; the
duplicate Wan `transformer/` is not used. The adapter's base model and those
three frozen modules must be available locally or in the Hugging Face cache.

The script defaults to dry-run. After checking predicted commands and timing,
add `--publish` to permit ROS command publication. The robot-side client rejects
targets outside the checkpoint's saved action min/max envelope. Arm targets
within 4 degrees of current telemetry are published unchanged; larger per-joint
deltas are clipped to 4 degrees rather than rejecting the entire action.
Grippers have independent target and step limits.

LingBot-VA is substantially larger than PI0.5 deployment checkpoints: its base
transformer is about 9.5 GB and the required Wan frozen modules are about
13.7 GB on disk. A 16 GB GPU and 16 GB system-RAM robot is below the comfortable
local-inference range; validate memory in dry-run before enabling publication.

### H200 remote inference

Run the stateful model service on one H200. Keep it bound to localhost and set
a random shared token:

```bash
export LINGBOT_REMOTE_TOKEN='<random-token>'
CUDA_VISIBLE_DEVICES=7 python /data/lingbot_remote/lingbot_va/serve_lingbot_va.py \
  --model-dir /data/train_outputs/lingbot_va_pick_up_water_bottles_quantile_v2_2xh200/checkpoints/001000/pretrained_model \
  --base-model /path/to/lerobot_lingbot_va_base_snapshot \
  --wan-model /path/to/robbyant_lingbot_va_base_snapshot \
  --host 127.0.0.1 --port 8765 --offline
```

If the SSH gateway permits TCP forwarding, expose that localhost service to the
robot LAN through the workstation (`192.168.8.35` in the current setup):

```bash
ssh -g -N -L 0.0.0.0:8765:127.0.0.1:8765 \
  -p 2222 'USER@H200_GATEWAY'
```

The current H200 gateway disables SSH TCP forwarding. In that case, keep an
authenticated multiplexing master and bridge ordinary SSH session channels;
the remote `nc` command is only a byte stream to the localhost service:

```bash
CONTROL_PATH=/tmp/lingbot_h200_control
ssh -M -N -o ControlMaster=yes -o ControlPersist=no \
  -o ControlPath="$CONTROL_PATH" -p 2222 'USER@H200_GATEWAY'

python deploy/common/ssh_session_relay.py \
  --control-path "$CONTROL_PATH" \
  --ssh-target 'USER@H200_GATEWAY' --ssh-port 2222 \
  --listen-host 0.0.0.0 --listen-port 8765
```

Keep both processes running. The relay opens a multiplexed SSH session for each
HTTP connection and therefore does not repeat password/MFA authentication.

Store the same token on the robot with owner-only permissions, then start the
client without `--publish`:

```bash
install -m 600 /dev/null /home/ubuntu/.config/lingbot_remote_token
# Write the token through a protected interactive editor.

python /home/ubuntu/deploy/lingbot_va/deploy_lingbot_va_remote.py \
  --interactive \
  --server-url http://192.168.8.35:8765
```

The full checkpoint exists only on the inference server. Its authenticated
health response supplies the robot client with the three camera keys, 16 action
channels, and saved physical action bounds; no model files are installed on the
robot.

The first request returns 12 actions. Subsequent requests upload one observed
keyframe per four executed actions and return 16 actions. ROS publication,
freshness checks, non-finite checks, and the `--max-arm-step` clip remain local.
Arm targets within the default 4-degree limit are published unchanged; larger
per-joint deltas are clipped to 4 degrees instead of rejecting the whole action.
A transport timeout or malformed chunk terminates the client without publishing
the pending command. Add `--publish` only after a successful dry-run.

### Jetson AGX Thor inference

The current Thor is managed over USB at `192.168.55.1` and serves the robot LAN
at `192.168.8.85`. Robot 283 is `192.168.8.42`. Build the pinned ARM64 inference
image from the runtime directory:

```bash
docker build -t lingbot-va-thor:0.6.0 \
  -f /home/wayne/lingbot_runtime/Autolife_VLA_Tools/deploy/lingbot_va/thor/Dockerfile \
  /home/wayne/lingbot_runtime
```

The model layout is:

```text
/home/wayne/lingbot_runtime/models/
  checkpoint/pretrained_model/  # QUANTILES v2 adapter only
  base/                          # lerobot/lingbot_va_base
  wan/vae/
  wan/text_encoder/
  wan/tokenizer/
```

Never place an `ACTION: IDENTITY` checkpoint at the production checkpoint path.
Start the authenticated service with:

```bash
/home/wayne/lingbot_runtime/Autolife_VLA_Tools/deploy/lingbot_va/thor/start_lingbot_va_server.sh
```

Validate from robot 283 without publishing:

```bash
python /home/ubuntu/deploy/lingbot_va/deploy_lingbot_va_remote.py \
  --server-url http://192.168.8.85:8765 \
  --token-file /home/ubuntu/.config/lingbot_remote_token \
  --task "pick up the bottle of water" \
  --max-steps 1
```

Only add `--publish` after inspecting the dry-run action envelope, inference
latency, current robot pose, camera freshness, and emergency-stop state.
