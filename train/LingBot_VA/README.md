# LingBot-VA 训练

以下命令使用 LeRobot 0.6.0，在物理 GPU 4、5 上进行双卡训练。
模型、数据集、输出目录、batch size、训练步数、保存频率和 W&B 等训练参数均由 YAML 配置。

## 训练新模型

启动前先确认 YAML 中的 `output_dir` 是一个新的、未被其他训练使用的目录；
checkpoint 将保存到该目录的 `checkpoints/` 下。

```bash
ENVROOT=/home/apps/data/Wayne/gjx/h200_transfer_20260716/conda_envs/lerobot-pi05-train
CONFIG=/data/Wayne/lwy/train/LingBot_VA/train_lingbot_va_pick_up_water_bottles_quantile_v2.yaml
TMPROOT=/data/Wayne/lwy/tmp/lingbot_fresh_v4
LOGDIR=/data/Wayne/lwy/logs
WANDBROOT=/data/Wayne/lwy/wandb

mkdir -p "$TMPROOT" "$LOGDIR" \
  "$WANDBROOT/data" "$WANDBROOT/cache" "$WANDBROOT/runs"

CUDA_VISIBLE_DEVICES=4,5 \
TMPDIR="$TMPROOT" \
HF_HOME=/data/Wayne/lwy/cache/huggingface \
TORCH_HOME=/data/Wayne/lwy/cache/torch \
WANDB_DATA_DIR="$WANDBROOT/data" \
WANDB_CACHE_DIR="$WANDBROOT/cache" \
WANDB_DIR="$WANDBROOT/runs" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "$ENVROOT/bin/accelerate" launch \
  --num_processes=2 \
  --main_process_port=29645 \
  "$ENVROOT/bin/lerobot-train" \
  --config_path="$CONFIG" \
  > "$LOGDIR/lingbot_va_quantile_v4_fresh_gpu45_bs32_steps5000_wandb.log" 2>&1 < /dev/null &

echo "training pid: $!"
```

## 从 checkpoint 接续训练

将 `CHECKPOINT` 改成要接续的 checkpoint。`--resume=true` 会恢复模型权重、
优化器、学习率调度器和训练 step。

```bash
ENVROOT=/home/apps/data/Wayne/gjx/h200_transfer_20260716/conda_envs/lerobot-pi05-train
CHECKPOINT=/data/train_outputs/lingbot_va_pick_up_water_bottles_quantile_v4_bs32_2xh200/checkpoints/000100/pretrained_model
TMPROOT=/data/Wayne/lwy/tmp/lingbot_resume_v4
LOGDIR=/data/Wayne/lwy/logs
WANDBROOT=/data/Wayne/lwy/wandb

mkdir -p "$TMPROOT" "$LOGDIR" \
  "$WANDBROOT/data" "$WANDBROOT/cache" "$WANDBROOT/runs"

CUDA_VISIBLE_DEVICES=4,5 \
TMPDIR="$TMPROOT" \
HF_HOME=/data/Wayne/lwy/cache/huggingface \
TORCH_HOME=/data/Wayne/lwy/cache/torch \
WANDB_DATA_DIR="$WANDBROOT/data" \
WANDB_CACHE_DIR="$WANDBROOT/cache" \
WANDB_DIR="$WANDBROOT/runs" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "$ENVROOT/bin/accelerate" launch \
  --num_processes=2 \
  --main_process_port=29645 \
  "$ENVROOT/bin/lerobot-train" \
  --config_path="$CHECKPOINT/train_config.json" \
  --resume=true \
  > "$LOGDIR/lingbot_va_quantile_v4_resume_gpu45.log" 2>&1 < /dev/null &

echo "training pid: $!"
```

`steps` 表示目标总步数。例如 checkpoint 为 `000200`、配置中的
`steps: 5000` 时，会从 step 200 接着训练到 step 5000，而不是再训练
5000 step。
