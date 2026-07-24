# AutoLife Isaac Sim Digital Twin

该工具订阅指定真实机器人的 ROS 2 全身状态，把控制器的 23 维关节角从度转换为
Isaac Sim/URDF 使用的弧度，并按关节名同步到 Isaac Sim。同步按关节名完成，而不是
依赖 Isaac Sim 内部 DOF 顺序；URDF 的夹爪 mimic 关节也会一起更新。

默认 URDF：

```text
/home/wayne-cb/Desktop/autolife_s1(1)/autolife_s1/urdfs/robot_v2_2.urdf
```

## 启动

机器人和本机 ROS 2/DDS 网络连通后执行：

```bash
cd /home/wayne-cb/Desktop/Autolife_VLA_Tools
chmod +x digital_twin/run_digital_twin.sh
digital_twin/run_digital_twin.sh --robot-id 283
```

连接另一台机器人只需更换 ID：

```bash
digital_twin/run_digital_twin.sh --robot-id 300
```

默认只打开 Isaac Sim 数字孪生窗口，避免图像编码或机器人视觉服务负载影响
关节状态链路。需要同时打开独立的 Ubuntu/OpenCV 三路 RGB 窗口时执行：

```bash
digital_twin/run_digital_twin.sh --robot-id 300 --camera-viewer
```

窗口从上到下依次显示头部彩色相机、左手相机和右手相机。相机查看器通过机器人
官方 SDK 的共享内存消费者读取现有图像，再经 SSH 传输 JPEG；只读图像，不会
向机器人发布动作控制消息。
按 `q`、`Esc` 或关闭相机窗口可单独退出相机查看器。Isaac Sim 主进程退出后，
相机查看器也会自动退出。同一台机器人只允许一个数字孪生相机查看器占用相机，
重复启动会给出明确错误而不是争用设备。

当前已知机器人地址：

```text
283 -> 192.168.8.42
300 -> 192.168.8.202
```

这两台机器人将自动使用 SSH 转发状态，因为机器人控制进程的 CycloneDDS
绑定在其本机 `127.0.0.1`。如果地址变化，可显式覆盖：

```bash
digital_twin/run_digital_twin.sh \
  --robot-id 300 \
  --robot-host 192.168.8.202
```

话题按下式自动生成：

```text
/topic_arm_whole_body_and_gripper_current_joints_status_<ROS_DOMAIN_ID>_<ROBOT_ID>
```

例如 `--ros-domain-id 0 --robot-id 283` 对应
`/topic_arm_whole_body_and_gripper_current_joints_status_0_283`。

常用参数：

```bash
# 指定 ROS domain
digital_twin/run_digital_twin.sh \
  --robot-id 283 \
  --ros-domain-id 0

# 完全覆盖状态话题
digital_twin/run_digital_twin.sh \
  --robot-id 283 \
  --state-topic /my/custom/status_topic

# 无窗口运行，或显示控制器目标而非实测关节
digital_twin/run_digital_twin.sh --robot-id 283 --headless
digital_twin/run_digital_twin.sh --robot-id 283 --source target

# 调整地面高度、大小和 RGB 颜色，或不创建地面
digital_twin/run_digital_twin.sh --robot-id 283 --ground-z -0.02 --ground-size 30
digital_twin/run_digital_twin.sh --robot-id 300 --ground-color 0.25 0.10 0.45
digital_twin/run_digital_twin.sh --robot-id 283 --no-ground

# 同时打开三路相机窗口
digital_twin/run_digital_twin.sh --robot-id 300 --camera-viewer

# 即使环境变量 SHOW_CAMERAS=1，也强制不打开相机窗口
digital_twin/run_digital_twin.sh --robot-id 300 --no-camera-viewer
```

默认地面颜色为蓝色 `RGB=(0.12, 0.32, 0.48)`，每个颜色分量范围是 `0~1`。

也可不启动 Isaac Sim，单独打开三路相机查看器：

```bash
digital_twin/run_camera_viewer.sh --robot-id 300
```

相机查看器常用参数：

```bash
digital_twin/run_camera_viewer.sh \
  --robot-id 300 \
  --fps 15 \
  --jpeg-quality 80 \
  --tile-width 400
```

`--rmw-implementation` 默认为 `auto`：已安装 CycloneDDS 时优先使用
`rmw_cyclonedds_cpp`，否则自动使用 `rmw_fastrtps_cpp`。两种 ROS 2 DDS
实现可以通信。只有选中 CycloneDDS 时才可使用
`--network-interface-address 192.168.8.38` 限定网卡。

如果必须使用 CycloneDDS，但系统尚未安装：

```bash
sudo apt install ros-jazzy-rmw-cyclonedds-cpp
digital_twin/run_digital_twin.sh \
  --robot-id 283 \
  --rmw-implementation rmw_cyclonedds_cpp \
  --network-interface-address 192.168.8.38
```

使用其他 Isaac Sim 安装位置时：

```bash
ISAAC_SIM_PATH=/path/to/isaacsim digital_twin/run_digital_twin.sh --robot-id 283
```

查看全部参数：

```bash
digital_twin/run_digital_twin.sh --help
```

## 行为和故障保护

- 默认 `--source measured`，因此数字孪生显示真实机器人实际达到的位置。
- 同一机器人只允许一个数字孪生实例，防止重复 Isaac 窗口造成误判。
- 三路相机窗口默认关闭；姿态同步确认正常后可用 `--camera-viewer` 单独启用。
- 这是姿态镜像而非自由动力学仿真；机器人固定底座且关闭重力，但仍以 60 Hz
  推进 PhysX，使关节张量可靠同步到视口。每帧在物理步进后、渲染前重新写入
  当前关节状态和 URDF drive target，避免无刚度/阻尼的关节发生动态漂移。
- 默认在 `z=0` 创建一块蓝色 `20 m × 20 m` 的可视化/碰撞地面。
- 收到不完整或非法 JSON 状态包时不会更新模型。
- 遥测中断超过 1 秒时发出警告，并将 Isaac Sim 模型冻结在最后有效姿态。
- 启动后没有状态时会持续等待，并每 5 秒打印一次正在等待的话题，不会自行关闭。
- 如需超时退出，可显式设置 `--initial-state-timeout-sec 15`。
- 本工具只订阅状态，不向真实机器人发布命令。

若 DDS 发现不到机器人，先用相同 ROS 环境确认话题：

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic echo --once \
  /topic_arm_whole_body_and_gripper_current_joints_status_0_283 \
  std_msgs/msg/String
```

运行纯 Python 映射测试（不需要启动 Isaac Sim）：

```bash
python3 -m unittest digital_twin.tests.test_joint_mapping -v
```
