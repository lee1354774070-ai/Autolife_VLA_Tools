# LeRobot Collector Usage

This tool records robot demonstrations as an official `LeRobotDataset`. Normal
collection uses one terminal: start the launcher and control episodes with the
keyboard.

## 1. Start

```bash
cd /home/ubuntu/lerobot_data_collector
bash ./start_lerobot_official_collect.sh <task_name> "<task text>"
```

Example:

```bash
TASK_TEXT="pick up the water bottle" \
bash ./start_lerobot_official_collect.sh pick_up_water_bottle
```

The default dataset path is:

```text
/home/ubuntu/nas14/<task_name>/dataset/
```

## 1.1 View parameter help

Show all launcher environment variables:

```bash
bash ./start_lerobot_official_collect.sh --help
```

Show one launcher variable:

```bash
bash ./start_lerobot_official_collect.sh --help MAX_SYNC_DELTA_SEC
bash ./start_lerobot_official_collect.sh MAX_SYNC_DELTA_SEC --help
```

Show all recorder command-line options:

```bash
python ./record_lerobot_official.py --help
```

Show one recorder option:

```bash
python ./record_lerobot_official.py --max-sync-delta-sec --help
python ./record_lerobot_official.py --with-depth --help
```

Single-option help does not require other mandatory arguments and never starts recording.

Change the storage root with:

```bash
OUTPUT_BASE_DIR=/mnt/data \
TASK_TEXT="put the water bottle in the box" \
bash ./start_lerobot_official_collect.sh put_water_bottle_in_box
```

## 2. Episode Controls

After the launcher prints `Interactive controls`, use the same terminal:

| Key | Action |
| --- | --- |
| `Enter` | Start a new episode |
| `S` / `s` | Save the current episode and pause |
| `D` / `d` | Discard the current episode and pause |
| `Q` / `q` | Save a valid episode and quit |
| `Ctrl+C` | Stop collection and clean up |

If synchronization fails, the terminal prints `EPISODE INVALID`. Press `D` to
discard it before starting another episode.

## 3. Common Configurations

### 30 FPS with head, waist, and depth

```bash
COLLECT_FPS=30 \
WITH_HEAD=1 \
WITH_WAIST=1 \
WITH_DEPTH=1 \
bash ./start_lerobot_official_collect.sh hotel_service
```

This records:

- 23-D `observation.state`;
- 23-D default action;
- head RGB, left-hand, right-hand, and depth cameras;
- `hand_left` as the default synchronization reference.

### Basic joints and RGB

```bash
bash ./start_lerobot_official_collect.sh rgb_test
```

### Head-color camera test

```bash
CAMERA_ONLY=1 bash ./start_lerobot_official_collect.sh camera_test
```

This mode uses state as the action fallback and is intended for camera-pipeline
checks only.

## 4. Common Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OUTPUT_BASE_DIR` | `/home/ubuntu/nas14` | Dataset parent directory |
| `TASK_TEXT` | task name | Task text stored in the dataset |
| `COLLECT_FPS` | `30` | Dataset recording FPS |
| `WITH_HEAD` | `0` | Add three neck joints |
| `WITH_WAIST` | `0` | Add four waist/leg joints |
| `WITH_DEPTH` | `0` | Add depth video |
| `ACTION_MODE` | `status_target` | `status_target`, `joint`, or `eef` |
| `IMAGE_SOURCE` | `shm` | Direct SHM or ROS image input |
| `IMAGE_POLL_FPS` | `120` | SHM polling rate, not dataset FPS |
| `SYNC_IMAGE_BUFFER_SIZE` | `16` | Image FIFO capacity per camera |
| `SYNC_REFERENCE_CAMERA` | `hand_left` | Synchronization reference camera |
| `MAX_SYNC_DELTA_SEC` | `0.03` | Maximum camera time delta |
| `MIN_CAMERAS` | `1` | Minimum number of live cameras |

Use the ROS image path with:

```bash
IMAGE_SOURCE=ros bash ./start_lerobot_official_collect.sh ros_test
```

## 5. Tasks and Dataset Resume

`task_name` is used as:

- the dataset directory name;
- the default local repo id;
- the task text when `TASK_TEXT` is not set.

Starting the same `task_name` again resumes the existing dataset when possible.
The camera set, FPS, depth setting, head/waist settings, and action mode must
remain unchanged.

Use a new task name or output directory when changing those features.

The launcher prints the cameras actually being recorded. Check this list before
collecting production data.

## 6. Output

```text
/home/ubuntu/nas14/<task_name>/
├── dataset/
│   ├── data/             # parquet data
│   ├── meta/             # LeRobot metadata
│   ├── videos/           # video files
│   └── sync_log.jsonl    # synchronization log
└── logs/                 # collection logs
```

Multiple video files are normal. Do not concatenate them before training.

## 7. Requirements

The robot should provide:

- ROS2 Jazzy;
- the `robot_env` environment;
- the `lerobot` environment;
- LeRobot 0.6 or newer for depth recording;
- running arm-state and camera services.

For missing state, missing cameras, or dataset configuration errors, check the
terminal output and the task directory's `logs/` files.

See [INSTRUCTION.md](INSTRUCTION.md) for implementation details.
