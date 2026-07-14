# Collector 实现说明

本文档介绍 Autolife LeRobot collector 的内部设计。日常启动命令和环境变量
请查看 [README_zh.md](README_zh.md)。

## 1. 设计约束

collector 遵循以下约束：

1. 每条保存的 episode 都属于一个合法的官方 `LeRobotDataset` root。
2. 每路图像独立缓存，并按照 FIFO 顺序处理。
3. 每个 dataset 行由一帧参考相机图像的时间戳锚定。
4. state 插值到图像锚点时间。
5. action 必须具有因果性，只能使用锚点时间或之前的信号。
6. 真实同步失败会使整条当前 episode 作废。
7. state 和关节 action 使用同一套 canonical schema 和索引顺序。
8. 相机及信号缓存有上限，内存不会无限增长。

collector 不会给机器人发送运动控制指令，只读取 state、target/action topic
和相机数据，并生成示范数据。

## 2. 模块职责

| 文件 | 职责 |
| --- | --- |
| `start_lerobot_official_collect.sh` | 解析配置、创建 IPC、按顺序启动进程并处理按键 |
| `record_lerobot_official.py` | 读取 ROS/SHM 数据、同步信号、构造 frame、管理 LeRobotDataset |
| `camera_config.py` | 定义相机名称、SHM 路径、ROS topic 和共享默认值 |
| `shm_camera.py` | 校验 metadata/图像数据并读取完整 SHM 帧 |
| `hand_camera_producer.py` | 解码手部相机 V4L2 MJPEG 并写入兼容 SDK 的 SHM |
| `shm_camera_topic_bridge.py` | 可选的 SHM 到 ROS Image bridge |
| `robot_schema.py` | 定义 canonical 关节组、名称、payload 字段和 action 拼接 |
| `time_sync.py` | 提供时间戳归一化、FIFO 就绪、最近匹配和插值函数 |
| `collector_control.py` | FIFO 命令、确认、摘要、相机检测和同步报告 |
| `tests/` | 测试配置、schema、时间同步、SHM 和控制协议 |

同步策略由 recorder 统一管理。相机进程只负责采集和发布图像，launcher 只
负责启动顺序和操作员控制。

## 3. 运行进程

默认 direct-SHM 模式：

```text
start_lerobot_official_collect.sh
├── official_lerobot_recorder      （LeRobot 环境）
└── hand_camera_producer           （robot 环境，需要时启动）
```

设置 `IMAGE_SOURCE=ros` 时增加：

```text
└── shm_camera_topic_bridge        （robot 环境 + ROS2）
```

recorder 需要 LeRobot、NumPy、ROS2 Python 和消息定义；producer 与 bridge 需要
机器人环境中的 OpenCV、设备和 ROS2 库。launcher 先 source
`/opt/ros/jazzy/setup.bash`，再为不同子进程设置各自的动态库路径、
`PYTHONPATH` 和 CycloneDDS 配置。

## 4. 启动和退出顺序

launcher 的启动顺序是：

1. 解析 task、输出目录、ROS 身份、feature 开关和日志序号。
2. 如果 PID 文件中的旧进程仍然存在，则拒绝重复启动。
3. 创建控制 FIFO 和临时 IPC 文件。
4. 在 collector 自有相机进程之前启动 recorder。
5. recorder 订阅 state/action topic，开始填充信号缓存。
6. 收到第一包完整 state 后，原子写入 state-ready 文件。
7. 按配置启动手部相机 producer。
8. direct 模式由 recorder 读取 SHM；ROS 模式启动 bridge。
9. 等待相机检测和数据集创建完成。
10. 处理采集终端中的键盘命令。

退出时，recorder 保存有效的 pending episode，丢弃无效 episode，finalize
encoder 和 metadata 后退出。launcher 按相反顺序停止配套进程，只删除临时
IPC 文件，数据集和日志都会保留。

## 5. 相机采集

### 5.1 SHM 格式

每路相机使用两个文件：

```text
/dev/shm/camera_metadata_struct_<name>
/dev/shm/camera_image_buffer_<name>
```

metadata 使用小端结构 `"<qiiiii"`：

```text
int64 timestamp_ns
int32 width
int32 height
int32 channels
int32 pixel_format
int32 byte_count
```

`shm_camera.read_shm_frame()` 先读取 metadata，检查尺寸和字节数，复制所需
图像字节，再次读取 metadata。如果第二次读取结果不同，说明 producer 在复制
期间更新了数据，当前这一代图像会被拒绝。

RGB 使用 pixel format `1`、3 个 `uint8` 通道，原始顺序为 BGR。depth 使用
pixel format `2`、1 个小端 `uint16` 通道。recorder 将 RGB 转为连续的 RGB HWC，
depth 保留为单通道 uint16 HWC。

某些机器人固件在 metadata 中提供的并不是 Unix 时间戳。`shm_timestamp_sec()`
只接受合理的 Unix 纳秒时间戳，否则回退到本地接收时间。这样可以避免不同
时间域直接比较；如果以后确认相机使用统一硬件时钟，还可以进一步提高实际
拍摄时刻同步精度。

### 5.2 direct-SHM 两阶段轮询

`IMAGE_POLL_FPS` 是 metadata 轮询频率，不是数据集帧率。默认 120Hz、数据集
30Hz 时，每个预期输出帧大约检查四次 metadata。

每次轮询分两个阶段：

1. 在一个很短的窗口内读取所有配置相机的 metadata；
2. 只复制时间戳发生变化的图像 buffer。

这样大尺寸 RGB/depth 复制不会阻塞其他相机的 metadata 发现。
`last_shm_timestamps` 用来防止重复帧。稳定性检查失败后，下一次轮询会重试。

SHM producer 只提供最新一帧，不提供历史帧队列。因此 direct-SHM 能减少传输
延迟并稳定观察 30 FPS，但不能恢复已经被 producer 覆盖的旧帧。

### 5.3 可选 ROS bridge

设置 `IMAGE_SOURCE=ros` 后，`shm_camera_topic_bridge.py` 读取同样的 SHM 文件，
使用 best-effort、depth=1 的 QoS 发布 `/camera/<name>/image_raw`。recorder
会根据 `msg.step` 正确处理行 padding，并使用同样的图像校验和同步逻辑。
ROS 模式适合兼容旧流程和诊断；direct 模式可以减少图像序列化、DDS 调度和
订阅回调排队。

## 6. 图像 FIFO 和匹配

recorder 为每路相机维护一个有界 `deque`。默认容量为：

```text
SYNC_IMAGE_BUFFER_SIZE=16
```

30 FPS 下约可缓存 533ms。state 和 action 使用
`SYNC_SIGNAL_BUFFER_SIZE=64` 的历史缓存。

ROS callback 和 direct-SHM 都调用公共的 `_store_image()`。该函数更新
`latest_images`，创建 `ImageSample`，并将图像加入对应相机 FIFO。如果 FIFO
已经满，会写入溢出事件；在正在录制 episode 时，该相机第一次溢出会以
`image_buffer_overflow:<camera>` 作废当前 episode。

每个录制 tick 的处理顺序：

1. 选择等待时间同步窗口结束、且比上一个锚点更新的最早参考相机帧；
2. 检查参考相机是否过旧或停止；
3. 检查参考帧间隔是否符合 `1 / COLLECT_FPS`；
4. 找到锚点前后的 state 样本；
5. 将 state 线性插值到锚点，并检查样本年龄和区间跨度；
6. 从每路其他相机 FIFO 中选择时间最近的图像；
7. 拒绝缺失、过旧或超过时间差阈值的图像；
8. 在锚点时间构造因果 action；
9. 将完整 frame 放入 LeRobot 的 pending episode buffer。

默认情况下，如果 `hand_left` 处于启用相机集合中，就使用它作为同步参考，
因为在 283 机器人的 30 FPS 验证中它的接收周期最稳定。可以通过
`SYNC_REFERENCE_CAMERA` 显式覆盖；camera-only 或自定义相机配置则回退到第一路
有效相机。

frame 成功加入 dataset 后，各路 FIFO 会删除本次匹配帧及其之前的旧帧。这样
FIFO 容量只用于吸收短时抖动，不会在正常录制过程中持续增长。

`oldest_ready_sample()` 保证参考帧按 FIFO 顺序消费，`nearest_sample()` 完成
跨相机最近匹配。程序不会静默删除一帧后把后续帧当作连续物理时间保存。

## 7. 同步失败策略

以下情况会作废当前 episode：

- 参考相机缺失、停止或过旧；
- 参考帧周期异常；
- state 插值区间缺失、过旧或跨度过大；
- 其他相机缺失、过旧或与锚点不同步；
- action 缺失、过旧或不满足因果性；
- 录制过程中图像 FIFO 溢出。

`no_reference_frame_ready` 表示最早参考帧仍在等待同步窗口，是临时 dropped
tick，不会单独导致 episode 作废。

同步日志是 JSONL。frame 事件包含锚点时间、帧间隔、state 插值信息、各路时间
差和本地接收年龄；drop 和 invalidation 事件包含原因及诊断值。
`collector_control.py sync-report` 会在不把全部 delta 载入内存的情况下统计
接受帧数、丢弃数、无效 episode、p95 和最大时间差。

## 8. 机器人 Schema 和 Action

`robot_schema.py` 是 canonical 关节顺序的唯一来源。固定 16 维前缀为左右两条
7 维手臂，后接左右夹爪。可选关节按以下顺序追加：

```text
WITH_HEAD=1:
  neck_roll, neck_pitch, neck_yaw

WITH_WAIST=1:
  leg_ankle, leg_knee, waist_pitch, waist_yaw
```

解析器会拒绝不完整关节组，不会静默为缺失关节填零。

`status_target` 从全身状态包的 target 字段读取目标。当前固件没有独立夹爪
target，因此夹爪 action 使用实测夹爪位置。`joint` 使用显式手臂和夹爪命令
topic，并按照相同 schema 拼接。`eef` 是独立的 15 维调试 action，名称故意
不与 state 相同。

## 9. Episode 和 Dataset 生命周期

按下 `Enter` 后，只记录比按键时参考时间更新的图像帧，并清空预热阶段的图像
FIFO，避免按键前的帧占用新 episode 容量或被写入。`S` 调用
`LeRobotDataset.save_episode()`；`D` 调用
`clear_episode_buffer(delete_images=True)`；`Q` 保存有效 pending episode、
丢弃无效 episode、finalize 数据集并退出。

新 root 会根据实际相机尺寸创建视频 feature，根据 schema 创建 state/action
feature。已有 root 在 resume 前会检查 feature 集合、尺寸、depth 模式和 schema。
摄像头集合、FPS、depth、头腰关节或 action mode 变化时必须使用新 root。

task text 会写入每帧和 episode metadata。launcher 使用 task name 作为目录名，
因此多个任务共用一个 root 时，必须显式设计共享 root 流程，并保证 feature
配置完全一致。

LeRobot 按相机、episode 或 chunk 拆分视频是正常行为。视频文件由 metadata
索引，训练前不要手动拼接。

## 10. IPC 和控制协议

launcher 与 recorder 在任务目录下共享以下临时文件：

```text
.official_recording_control       命令 named pipe
.official_recording_status.json   save/discard 确认
.official_recorder_state_ready.json
.official_episode_event.json
.official_recording_pids
```

命令按行传输：`start`、`save`、`discard`、`quit`，可附带 request ID。正常
用户界面是 launcher 中的键盘循环。状态 JSON 使用 `os.replace` 原子写入，
launcher 不会读到半个 JSON 文件。Python IPC 层仍然独立保留，避免把进程
控制逻辑重复写进 shell 脚本。

## 11. 安全扩展位置

新增相机时，先修改 `camera_config.py`，再更新 launcher 相机选择和测试。
新增关节组时，修改 `robot_schema.py` 和 schema 测试，不要在 recorder 中重复
维护名称。新增信号时，应创建带时间戳的 buffer 和明确的因果选择规则。

恢复已有 root 时不要改变 feature shape；不要用 latest-frame 覆盖替代 FIFO
匹配；也不要只为了减少 invalid episode 而放宽同步阈值。

## 12. 验证

使用 LeRobot 环境运行纯测试：

```bash
/home/wayne-cb/miniconda3/envs/lerobot/bin/python -m unittest discover -s tests -v
python3 -m py_compile *.py
bash -n *.sh
```

机器人现场还需要检查 ROS discovery、SHM 文件、V4L2 设备、实际相机 FPS、
FFmpeg 编码器、CPU 和磁盘吞吐。无动作验证时应打开生产环境的所有相机和
关节组，使用正式 FPS 运行，并检查 `sync_log.jsonl` 中的 drop 和 invalidation。
