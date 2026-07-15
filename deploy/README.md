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
