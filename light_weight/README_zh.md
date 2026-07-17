# PI0.5 模型量化

本目录用于生成独立的 TorchAO 权重量化 PI0.5 模型，不会修改原始 LeRobot
checkpoint。

## 安装

进入运行 LeRobot 的同一个环境，再安装与当前 PyTorch 匹配的 TorchAO：

```bash
conda activate lerobot
pip install torchao
```

## 量化

建议先测试 INT8 weight-only。它只量化较大的 Linear 权重，激活值仍保持模型原有
精度，并默认保留最终动作输出层的原始精度。

```bash
cd /home/ubuntu/Autolife_VLA_Tools/light_weight

python quantize_pi05.py \
  --model-dir /mnt/nas14/pi05_models/MZJ/pi05_baseline_003000_pretrained_model/pretrained_model \
  --output-dir /mnt/nas14/pi05_models/MZJ/pi05_baseline_003000_int8wo \
  --method int8wo \
  --device cuda \
  --verify-load
```

只检查将量化哪些层，不写入文件：

```bash
python quantize_pi05.py --model-dir MODEL --dry-run --list-modules
```

实验性 INT4 weight-only：

```bash
python quantize_pi05.py \
  --model-dir MODEL \
  --output-dir OUTPUT \
  --method int4wo \
  --group-size 128 \
  --verify-load
```

可重复使用 `--exclude REGEX` 将指定模块保留为 BF16。只有验证动作精度后才建议使用
`--quantize-action-output`。执行 `python quantize_pi05.py --help` 可查看全部参数。

## 输出与加载

输出目录包含原模型的 processor/config 文件、`quantized_state_dict.pt` 和
`quantization_manifest.json`，不会复制浮点 `model.safetensors`。

量化权重不能直接由原版 `PI05Policy.from_pretrained()` 加载，需使用配套加载器：

```python
from light_weight import load_quantized_policy

policy = load_quantized_policy(
    "/path/to/pi05_int8wo",
    device="cuda",
    compile_mode="max-autotune-no-cudagraphs",
)
```

`.pt` 反序列化只适用于自己生成或可信来源的模型。量化会改变模型数值，控制机器人前
需要对比原模型和量化模型的峰值显存、chunk 推理耗时、动作误差及真实任务成功率。

使用仓库中的量化模型专用入口部署：

```bash
python ../deploy/deploy_pi05_light_weight.py \
  --interactive \
  --model-dir /path/to/pi05_int8wo \
  --tokenizer-dir /path/to/paligemma-tokenizer \
  --dry-run
```

脚本默认使用 `max-autotune-no-cudagraphs`，并会在启动机器人控制前检查 PyTorch、
TorchAO 和 LeRobot 版本是否与量化 manifest 兼容。
