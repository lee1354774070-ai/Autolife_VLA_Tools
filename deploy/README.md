# Local PI0.5 Deployment

## Start

Run this on the robot after starting the robot control service and sourcing ROS2:

```bash
source /opt/ros/jazzy/setup.bash
conda activate lerobot
export ROS_DOMAIN_ID=0
export ROBOT_ID=283
export MODEL_DIR=/home/ubuntu/your_model/pretrained_model

python /home/ubuntu/deploy/deploy_pi05.py \
  --task "pick up the water bottle"
```

`MODEL_DIR` can be replaced by `--model-dir`; the command-line option takes
precedence. The default in the repository remains the old local
`004500/pretrained_model` path for compatibility.

The default observation contains the collector's three RGB features:

```text
observation.images.rgbd_head_color
observation.images.hand_left
observation.images.hand_right
```

The default policy controls the two arms and grippers only. The head and waist
remain at their latest measured positions in outgoing whole-body messages.

The checkpoint must have the same LeRobot input feature names and shapes. The
old 004500 checkpoint contains only `observation.images.rgbd_head_color`, so
run it in compatibility mode when needed:

```bash
python /home/ubuntu/deploy/deploy_pi05.py \
  --model-dir /home/ubuntu/004500/pretrained_model \
  --camera rgbd_head_color \
  --task "pick up the water bottle"
```

For a new three-camera checkpoint, omit `--camera` and the three default RGB
features are used.

Show all deployment parameters or one parameter:

```bash
python /home/ubuntu/deploy/deploy_pi05.py --help
python /home/ubuntu/deploy/deploy_pi05.py --with-head --help
```

## Optional head and waist control

The policy must have been trained with the same dimensions as the selected
switches. The collector order is base 16, head 3, waist 4, giving 16, 19, 20,
or 23 dimensions:

```bash
# Base arms and grippers: 16-D
python /home/ubuntu/deploy/deploy_pi05.py --no-with-head --no-with-waist

# Arms, grippers, and head: 19-D
python /home/ubuntu/deploy/deploy_pi05.py --with-head --no-with-waist

# Arms, grippers, and waist: 20-D
python /home/ubuntu/deploy/deploy_pi05.py --no-with-head --with-waist

# Full body: 23-D
python /home/ubuntu/deploy/deploy_pi05.py --with-head --with-waist
```

`WITH_HEAD=1` and `WITH_WAIST=1` can be used instead of the command-line
switches. A 16-D 004500 checkpoint cannot be used with either optional switch;
train a matching checkpoint first.

## Camera and depth options

Select RGB cameras by repeating `--camera`. The names match the collector and
LeRobot feature keys:

```bash
python /home/ubuntu/deploy/deploy_pi05.py \
  --camera rgbd_head_color \
  --camera hand_left \
  --camera hand_right
```

Enable the collector's depth feature:

```bash
python /home/ubuntu/deploy/deploy_pi05.py \
  --with-depth \
  --camera rgbd_head_color \
  --camera hand_left \
  --camera hand_right
```

This adds `observation.images.rgbd_head_depth`. The depth model must have been
trained with that feature. Live uint16 millimetre depth is quantized with the
same `depth-min`, `depth-max`, `depth-shift`, and logarithmic settings used by
the collector.

The canonical joint order is the collector order: base 16 dimensions first,
then head 3 dimensions, then waist 4 dimensions. The script uses the standard
LeRobot keys `observation.state`, `observation.images.<camera>`, and `task`;
custom camera selection uses the same camera names defined by the collector.

## Test before moving the robot

Use `--dry-run` to load the model and execute inference without publishing
commands:

```bash
python /home/ubuntu/deploy/deploy_pi05.py \
  --model-dir /home/ubuntu/004500/pretrained_model \
  --task "pick up the water bottle" \
  --dry-run --max-steps 30
```

Press `Ctrl-C` to stop normal deployment. The script reads
the selected cameras directly from `/dev/shm`, matches the collector's
metadata-batch synchronization and BGR-to-RGB conversion, and publishes the
same body and gripper topics used by the collector.
