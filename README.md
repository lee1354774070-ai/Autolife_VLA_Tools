# Autolife VLA Tools

Tools for collecting LeRobot datasets, running local PI0.5 policies, and
building lightweight PI0.5 inference artifacts on an Autolife robot.

| Directory | Purpose |
| --- | --- |
| `lerobot_data_collector/` | Records synchronized robot state, actions, RGB, and optional depth into `LeRobotDataset`. |
| `deploy/` | Runs a local PI0.5 checkpoint with the same camera and joint schema. |
| `light_weight/` | Creates loadable TorchAO INT8 or INT4 weight-only PI0.5 artifacts. |

Start with the usage guide in the relevant directory:

- [Collector usage](lerobot_data_collector/README.md) / [中文](lerobot_data_collector/README_zh.md)
- [Deployment usage](deploy/README.md)
- [Quantization usage](light_weight/README.md) / [中文](light_weight/README_zh.md)

The collector implementation and synchronization guarantees are documented in
[INSTRUCTION.md](lerobot_data_collector/INSTRUCTION.md) / [中文](lerobot_data_collector/INSTRUCTION_zh.md).
