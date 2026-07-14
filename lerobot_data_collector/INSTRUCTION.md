# Collector Implementation Guide

This document describes the internal design of the Autolife LeRobot collector.
For daily commands and environment variables, see [README.md](README.md).

## 1. Design Invariants

The collector is built around these invariants:

1. Every saved episode belongs to one valid official `LeRobotDataset` root.
2. Image streams are buffered per camera in FIFO order.
3. A dataset row is anchored by one reference-camera timestamp.
4. State is linearly interpolated to the image timestamp.
5. Action is causal: it comes from the image timestamp or an earlier signal.
6. A real synchronization failure invalidates the whole current episode.
7. State and joint action use one canonical schema and one index order.
8. Camera and signal buffers are bounded; memory usage cannot grow forever.

The collector does not command robot motion. It observes state, target/action
topics, and camera streams, then stores the resulting demonstration.

## 2. Module Responsibilities

| File | Responsibility |
| --- | --- |
| `start_lerobot_official_collect.sh` | Resolve configuration, create IPC, launch processes, and handle keyboard input |
| `record_lerobot_official.py` | Read ROS/SHM data, synchronize streams, build frames, and manage LeRobotDataset |
| `camera_config.py` | Define camera names, SHM paths, ROS topics, and shared defaults |
| `shm_camera.py` | Validate metadata/image pairs and read complete SHM frames |
| `hand_camera_producer.py` | Decode hand-camera V4L2 MJPEG and write SDK-compatible SHM files |
| `shm_camera_topic_bridge.py` | Optional SHM-to-ROS Image bridge |
| `robot_schema.py` | Define canonical joint groups, names, payload keys, and action composition |
| `time_sync.py` | Provide timestamp normalization, FIFO readiness, nearest matching, and interpolation helpers |
| `collector_control.py` | Implement FIFO commands, acknowledgements, summaries, camera detection, and reports |
| `tests/` | Test pure configuration, schema, timing, SHM, and control behavior |

The recorder intentionally owns the synchronization policy. Camera producers
only acquire/publish images, and the launcher only manages process order and
operator control.

## 3. Runtime Processes

Default direct-SHM mode:

```text
start_lerobot_official_collect.sh
├── official_lerobot_recorder      (LeRobot environment)
└── hand_camera_producer           (robot environment, when needed)
```

Optional ROS mode adds:

```text
└── shm_camera_topic_bridge        (robot environment + ROS2)
```

The recorder needs LeRobot, NumPy, ROS2 Python, and the message definitions. The
producer and bridge need the robot environment, OpenCV, and ROS2 libraries. The
launcher sources `/opt/ros/jazzy/setup.bash`, then gives each child its own
`LD_LIBRARY_PATH`, `PYTHONPATH`, and CycloneDDS configuration.

## 4. Startup and Shutdown

The launcher follows this order:

1. Resolve task, output root, ROS identity, feature switches, and log sequence.
2. Reject a session whose PID file still contains a live process.
3. Create the control FIFO and transient IPC files.
4. Start the recorder before collector-owned camera processes.
5. Let the recorder subscribe to state/action topics and fill signal buffers.
6. After the first complete state packet, write the atomic state-ready marker.
7. Start the hand producer if requested and available.
8. In direct mode, let the recorder poll SHM. In ROS mode, start the bridge.
9. Wait for camera discovery and dataset creation.
10. Process keyboard commands from the collector terminal.

On quit, the recorder saves a valid pending episode, discards an invalid one,
finalizes encoders and metadata, and exits. The launcher then stops companion
processes in reverse launch order and removes only transient IPC files. Dataset
data and logs are preserved.

## 5. Camera Acquisition

### 5.1 SHM ABI

Each camera uses two files:

```text
/dev/shm/camera_metadata_struct_<name>
/dev/shm/camera_image_buffer_<name>
```

Metadata uses the little-endian structure `"<qiiiii"`:

```text
int64 timestamp_ns
int32 width
int32 height
int32 channels
int32 pixel_format
int32 byte_count
```

`shm_camera.read_shm_frame()` reads metadata, validates dimensions and byte
count, copies exactly the required image bytes, and reads metadata again. If
the second metadata value differs, the producer updated the pair during the
copy and the reader rejects that generation.

RGB uses pixel format `1`, three `uint8` channels, and BGR byte order. Depth
uses pixel format `2`, one little-endian `uint16` channel. The recorder converts
RGB to contiguous RGB HWC and preserves depth as uint16 HWC with one channel.

Some robot firmware versions expose a non-Unix timestamp in the metadata field.
`shm_timestamp_sec()` accepts a plausible Unix nanosecond timestamp and falls
back to local receive time otherwise. This keeps clocks in one domain, but a
confirmed common camera clock would provide better physical capture-time sync.

### 5.2 Direct-SHM Two-Phase Poll

`IMAGE_POLL_FPS` is the metadata polling rate, not the dataset FPS. At the
default 120Hz and dataset rate 30Hz, the recorder checks metadata about four
times per expected output frame.

Each poll has two phases:

1. Read metadata for all configured cameras in a tight pass.
2. Copy image buffers only for cameras whose timestamp is new.

This prevents a large RGB/depth copy from delaying metadata discovery for the
other cameras. `last_shm_timestamps` prevents duplicate frames. A failed
stable-pair read is retried on the next poll.

The SHM producer exposes only its latest frame, not a historical kernel queue.
The recorder can therefore reduce transport latency and observe 30 FPS reliably,
but it cannot recover a frame that the producer already overwrote.

### 5.3 Optional ROS Bridge

With `IMAGE_SOURCE=ros`, `shm_camera_topic_bridge.py` reads the same SHM files
and publishes `/camera/<name>/image_raw` using best-effort QoS and depth one.
The recorder converts ROS row padding correctly using `msg.step` and applies the
same image validation and synchronization logic. ROS mode is useful for
compatibility and diagnostics, while direct mode removes image serialization,
DDS scheduling, and subscriber callback queuing.

## 6. Image FIFO and Matching

The recorder owns one bounded `deque` per camera. The default capacity is 16:

```text
SYNC_IMAGE_BUFFER_SIZE=16
```

At 30 FPS this represents about 533ms per camera. `SYNC_SIGNAL_BUFFER_SIZE=64`
is used for state and action history.

The common `_store_image()` path is used by both ROS callbacks and direct SHM.
It updates `latest_images`, builds an `ImageSample`, and appends it to the
camera FIFO. If the bounded FIFO is already full, the event is logged. During
an active episode, the first overflow for that camera invalidates the episode
with reason `image_buffer_overflow:<camera>`.

At each recording tick, the recorder:

1. Selects the oldest reference-camera sample that has waited for the matching
   window and is newer than the previous anchor.
2. Rejects a stale or stalled reference camera.
3. Checks the reference-frame interval against `1 / COLLECT_FPS`.
4. Finds state samples immediately before and after the anchor.
5. Linearly interpolates state to the anchor and checks sample age/span.
6. Selects the nearest sample from every other camera FIFO.
7. Rejects a missing, stale, or over-delta camera sample.
8. Builds a causal action at or before the anchor.
9. Adds one complete frame to LeRobot's pending episode buffer.

After a frame is successfully added, the matched sample and all older samples
are removed from each camera FIFO. This makes FIFO capacity absorb temporary
bursts instead of growing during normal recording.

`oldest_ready_sample()` preserves FIFO order. `nearest_sample()` performs the
cross-camera lookup. The code never silently removes one image row and saves
later rows as if the physical time were continuous.

## 7. Synchronization Failure Policy

The following conditions invalidate the current episode:

- missing or stalled reference camera;
- unexpected reference-frame interval;
- missing or stale state interpolation bracket;
- missing, stale, or unsynchronized secondary camera;
- stale or non-causal action;
- image FIFO overflow during recording.

`no_reference_frame_ready` is different: it is a transient wait while the
oldest reference sample matures. It is reported as a dropped tick but does not
by itself invalidate the episode.

The synchronization log is JSONL. Frame events contain anchor timestamp,
interval, state interpolation information, per-stream timestamp deltas, and
local receive ages. Drop and invalidation events contain their reasons and
diagnostic values. `collector_control.py sync-report` summarizes accepted
frames, drops, invalid episodes, p95 deltas, and maxima without loading every
delta into memory.

## 8. Robot Schema and Actions

`robot_schema.py` is the only owner of canonical joint order. The fixed 16-D
prefix is two 7-D arms followed by left and right gripper values. Optional
groups append in this order:

```text
WITH_HEAD=1:
  neck_roll, neck_pitch, neck_yaw

WITH_WAIST=1:
  leg_ankle, leg_knee, waist_pitch, waist_yaw
```

The parser rejects incomplete groups and never silently fills a missing joint.

`status_target` reads target fields from the atomic whole-body status packet.
Because that packet has no independent gripper target in the current firmware,
the gripper action values use measured gripper positions. `joint` composes
explicit arm and gripper command topics using the same schema. `eef` is a
separate 15-D debug action and intentionally does not share state names.

## 9. Episode and Dataset Lifecycle

`Enter` records only frames newer than the reference timestamp observed at the
keypress. Warmup image FIFOs are cleared at that boundary, so pre-keypress
frames cannot consume the new episode's capacity or be written into it. `S` calls `LeRobotDataset.save_episode()`. `D` calls
`clear_episode_buffer(delete_images=True)`. `Q` saves a valid pending episode,
discards an invalid one, finalizes the dataset, and exits.

The default synchronization reference is `hand_left` when that camera is
active, because it had the most stable 30 FPS receive cadence in robot 283
validation. `SYNC_REFERENCE_CAMERA` can override it. Camera-only or custom
configurations fall back to their first active camera unless explicitly set.

For a new root, live camera dimensions create the video features and the schema
creates state/action features. For an existing root, feature set, shape, depth
mode, and schema are validated before resume. A changed camera set, FPS, depth
setting, head/waist setting, or action mode requires a new root.

The task text is written into each frame and episode metadata. The launcher uses
the task name as the directory name, so a multi-task root requires an
intentional shared-root workflow with identical feature configuration.

LeRobot may split videos by camera, episode, or chunk. These files are indexed
by the dataset metadata and must not be concatenated manually.

## 10. IPC and Control

The launcher and recorder share these transient files under the task directory:

```text
.official_recording_control       named pipe for commands
.official_recording_status.json   save/discard acknowledgement
.official_recorder_state_ready.json
.official_episode_event.json
.official_recording_pids
```

Commands are newline-delimited: `start`, `save`, `discard`, and `quit`, with an
optional request ID. The normal user interface is the keyboard loop in the
launcher. Status documents are written atomically with `os.replace` so the
launcher never reads a partial JSON file. The Python IPC layer remains separate
so process control is not duplicated inside the shell script.

## 11. Safe Extension Points

When adding a camera, update `camera_config.py`, then update the launcher camera
selection and tests. When adding a joint group, update `robot_schema.py` and
schema tests; do not duplicate names in the recorder. When adding a signal,
create a timestamped buffer and a causal selection rule in the recorder.

Do not change feature shape while resuming an existing root. Do not replace FIFO
matching with latest-frame reuse, and do not relax synchronization thresholds
just to make invalid episodes disappear.

## 12. Validation

Run pure tests with the LeRobot environment:

```bash
/home/wayne-cb/miniconda3/envs/lerobot/bin/python -m unittest discover -s tests -v
python3 -m py_compile *.py
bash -n *.sh
```

Robot-only validation must additionally check ROS discovery, SHM files, V4L2
devices, actual source FPS, video encoder availability, CPU load, and disk
throughput. A no-motion test should enable all intended cameras and joint
groups, run at the production FPS, and inspect `sync_log.jsonl` for drops and
episode invalidations.
