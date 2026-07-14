# Local PI0.5 Deployment

## Start

Run this on the robot after starting the robot control service and sourcing ROS2:

```bash
source /opt/ros/jazzy/setup.bash
conda activate lerobot
export ROS_DOMAIN_ID=0
export ROBOT_ID=283

python /home/ubuntu/deploy/deploy_pi05_004500.py \
  --model-dir /home/ubuntu/004500/pretrained_model \
  --task "pick up the water bottle"
```

The default policy controls the two arms and grippers only. The head and waist
remain at their latest measured positions in outgoing whole-body messages.

Show all deployment parameters or one parameter:

```bash
python /home/ubuntu/deploy/deploy_pi05_004500.py --help
python /home/ubuntu/deploy/deploy_pi05_004500.py --with-head --help
```

## Optional head and waist control

The policy must have been trained with the same dimensions as the selected
switches. The collector order is base 16, head 3, waist 4, giving 16, 19, 20,
or 23 dimensions:

```bash
# Base arms and grippers: 16-D
python /home/ubuntu/deploy/deploy_pi05_004500.py --no-with-head --no-with-waist

# Arms, grippers, and head: 19-D
python /home/ubuntu/deploy/deploy_pi05_004500.py --with-head --no-with-waist

# Arms, grippers, and waist: 20-D
python /home/ubuntu/deploy/deploy_pi05_004500.py --no-with-head --with-waist

# Full body: 23-D
python /home/ubuntu/deploy/deploy_pi05_004500.py --with-head --with-waist
```

`WITH_HEAD=1` and `WITH_WAIST=1` can be used instead of the command-line
switches. A 16-D 004500 checkpoint cannot be used with either optional switch;
train a matching checkpoint first.

## Test before moving the robot

Use `--dry-run` to load the model and execute inference without publishing
commands:

```bash
python /home/ubuntu/deploy/deploy_pi05_004500.py \
  --model-dir /home/ubuntu/004500/pretrained_model \
  --task "pick up the water bottle" \
  --dry-run --max-steps 30
```

Press `Ctrl-C` to stop normal deployment. The script reads
`rgbd_head_color` directly from `/dev/shm`, matches the collector's BGR-to-RGB
conversion and 480x640 model input, and publishes the same body and gripper
topics used by the collector.
