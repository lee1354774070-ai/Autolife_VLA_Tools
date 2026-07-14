# LeRobot 数据采集工具使用说明

本工具在机器人上采集数据，并保存为官方 `LeRobotDataset`。正常采集只需要
使用一个终端运行启动脚本并通过键盘控制。

## 1. 启动

```bash
cd /home/ubuntu/lerobot_data_collector
bash ./start_lerobot_official_collect.sh <task_name> "<task text>"
```

示例：

```bash
TASK_TEXT="pick up the water bottle" \
bash ./start_lerobot_official_collect.sh pick_up_water_bottle
```

默认数据保存到：

```text
/home/ubuntu/nas14/<task_name>/dataset/
```

修改保存位置：

```bash
OUTPUT_BASE_DIR=/mnt/data \
TASK_TEXT="put the water bottle in the box" \
bash ./start_lerobot_official_collect.sh put_water_bottle_in_box
```

## 2. 录制控制

启动完成并显示 `Interactive controls` 后，在当前终端使用：

| 按键 | 功能 |
| --- | --- |
| `Enter` | 开始录制一条新的 episode |
| `S` / `s` | 保存当前 episode，并暂停 |
| `D` / `d` | 丢弃当前 episode，并暂停 |
| `Q` / `q` | 保存有效 episode 并退出 |
| `Ctrl+C` | 停止采集并执行清理 |

如果某条 episode 发生同步失败，终端会提示 `EPISODE INVALID`。此时必须
按 `D` 丢弃，不能保存该 episode。

## 3. 常用采集配置

### 30 FPS、头部、腰部和 depth

```bash
COLLECT_FPS=30 \
WITH_HEAD=1 \
WITH_WAIST=1 \
WITH_DEPTH=1 \
bash ./start_lerobot_official_collect.sh hotel_service
```

此时：

- `observation.state` 为 23 维；
- 默认 action 为 23 维；
- 录制头部 RGB、左手、右手和 depth 四路相机；
- 默认使用 `hand_left` 作为同步参考相机。

### 只采集基础关节和 RGB

```bash
bash ./start_lerobot_official_collect.sh rgb_test
```

### 只测试头部彩色相机

```bash
CAMERA_ONLY=1 \
bash ./start_lerobot_official_collect.sh camera_test
```

该模式使用 state 作为 action 回退，只适合测试相机链路。

## 4. 常用环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `OUTPUT_BASE_DIR` | `/home/ubuntu/nas14` | 数据保存父目录 |
| `TASK_TEXT` | task name | 写入数据集的任务文本 |
| `COLLECT_FPS` | `30` | 数据集录制帧率 |
| `WITH_HEAD` | `0` | 增加 3 个头部关节 |
| `WITH_WAIST` | `0` | 增加 4 个腰部/腿部关节 |
| `WITH_DEPTH` | `0` | 增加 depth 视频 |
| `ACTION_MODE` | `status_target` | `status_target`、`joint` 或 `eef` |
| `IMAGE_SOURCE` | `shm` | `shm` 直读或 `ros` topic 输入 |
| `IMAGE_POLL_FPS` | `120` | SHM 检查频率，不是录制帧率 |
| `SYNC_IMAGE_BUFFER_SIZE` | `16` | 每路图像 FIFO 容量 |
| `SYNC_REFERENCE_CAMERA` | `hand_left` | 同步参考相机 |
| `MAX_SYNC_DELTA_SEC` | `0.03` | 相机最大同步时间差 |
| `MIN_CAMERAS` | `1` | 最少有效相机数量 |

例如，使用 ROS 图像 topic：

```bash
IMAGE_SOURCE=ros bash ./start_lerobot_official_collect.sh ros_test
```

## 5. 任务和数据集

`task_name` 同时用于：

- 数据集目录名称；
- 默认 repo id；
- 没有设置 `TASK_TEXT` 时的任务文本。

退出后再次使用相同的 `task_name`，工具会尝试继续写入原来的 dataset。以下
配置必须保持一致：相机集合、FPS、depth、头部/腰部关节和 action mode。

如果需要改变这些配置，请使用新的 `task_name` 或新的输出目录。

每次启动后，终端会打印实际录制的相机列表。正式采集前请确认列表正确。

## 6. 输出文件

```text
/home/ubuntu/nas14/<task_name>/
├── dataset/
│   ├── data/             # parquet 数据
│   ├── meta/             # LeRobot metadata
│   ├── videos/           # 视频文件
│   └── sync_log.jsonl    # 同步日志
└── logs/                 # 本次采集日志
```

视频被拆分成多个文件是正常的，不需要手动拼接后再训练。

## 7. 环境要求

机器人上需要准备：

- ROS2 Jazzy；
- `robot_env` 环境；
- `lerobot` 环境；
- LeRobot 0.6 或更高版本用于 depth 录制；
- 正常运行的手臂状态服务和相机服务。

如果提示没有完整 state、相机缺失或 dataset 配置不匹配，请先检查终端输出
和对应任务目录下的 `logs/` 文件。

详细实现说明请查看 [INSTRUCTION_zh.md](INSTRUCTION_zh.md)。
