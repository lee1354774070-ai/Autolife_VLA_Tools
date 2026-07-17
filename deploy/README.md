# Local PI0.5 Deployment

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

python /home/ubuntu/deploy/deploy_pi05.py \
  --task "pick up the water bottle"
```

`--model-dir` overrides `MODEL_DIR`. The repository default remains the legacy
`004500/pretrained_model` path for compatibility.

## Persistent interactive session

Use `--interactive` when the same model should remain in GPU memory across
multiple tasks. The process creates ROS2 and SHM readers immediately, but it
does not load the PI0.5 weights until `enable` is entered.

```bash
python /home/ubuntu/deploy/deploy_pi05.py \
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
python deploy_pi05.py --interactive --n-action-steps 10
```

At `--hz 10`, `--n-action-steps 10` requests a new chunk after about one
second of executed actions. Inference itself blocks the non-RTC control loop.

Enable Real-Time Chunking when inference should run in a background thread
while the control loop continues consuming the previous chunk:

```bash
python deploy_pi05.py \
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
python deploy_pi05.py --model-dir /home/ubuntu/004500/pretrained_model \
  --camera rgbd_head_color --task "pick up the water bottle"

# New default: 16-D joints and three RGB cameras.
python deploy_pi05.py --model-dir /home/ubuntu/new_model/pretrained_model

# 19-D: append head joints.
python deploy_pi05.py --with-head

# 20-D: append waist joints.
python deploy_pi05.py --with-waist

# 23-D: head and waist.
python deploy_pi05.py --with-head --with-waist

# Add the collector-compatible head depth feature.
python deploy_pi05.py --with-depth
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
python deploy_pi05.py --dry-run --max-steps 30
python deploy_pi05.py --help
python deploy_pi05.py --with-depth --help
```

The deployer reads images directly from `/dev/shm`, converts BGR to model RGB,
and publishes the same whole-body and gripper topics used by the collector.

## Quantized PI0.5

Install the TorchAO version recorded in the quantized checkpoint manifest, then
use the dedicated entry point:

```bash
python /home/ubuntu/Autolife_VLA_Tools/deploy/deploy_pi05_light_weight.py \
  --interactive \
  --model-dir /path/to/pretrained_model_int8wo \
  --tokenizer-dir /home/ubuntu/pi05_assets/paligemma-3b-pt-224 \
  --dry-run
```

It supports the same cameras, joint options, safety checks, and interactive
commands as `deploy_pi05.py`. Its default compile mode is
`max-autotune-no-cudagraphs`; override it with `--compile-mode disabled` when
debugging eager PyTorch.

The loader rejects incompatible PyTorch, TorchAO, or LeRobot versions before
initializing robot control. Re-quantize the model in the deployment environment
when the recorded and installed versions differ.
