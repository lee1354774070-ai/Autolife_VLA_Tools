# PI0.5 Quantization

This directory creates a separate TorchAO weight-only PI0.5 artifact. It never
modifies the source LeRobot checkpoint.

## Install

Activate the same environment used by LeRobot, then install the TorchAO build
compatible with that PyTorch installation:

```bash
conda activate lerobot
pip install torchao
```

## Quantize

INT8 weight-only is the recommended first test. It keeps activations in the
model's original precision and leaves the final action projection unquantized.

```bash
cd /home/ubuntu/Autolife_VLA_Tools/light_weight

python quantize_pi05.py \
  --model-dir /mnt/nas14/pi05_models/MZJ/pi05_baseline_003000_pretrained_model/pretrained_model \
  --output-dir /mnt/nas14/pi05_models/MZJ/pi05_baseline_003000_int8wo \
  --method int8wo \
  --device cuda \
  --verify-load
```

Inspect the selected layers without writing an artifact:

```bash
python quantize_pi05.py --model-dir MODEL --dry-run --list-modules
```

Experimental INT4 weight-only:

```bash
python quantize_pi05.py \
  --model-dir MODEL \
  --output-dir OUTPUT \
  --method int4wo \
  --group-size 128 \
  --verify-load
```

Use `--exclude REGEX` repeatedly to keep additional modules in BF16. Use
`--quantize-action-output` only after validating action accuracy. Run
`python quantize_pi05.py --help` for every option.

## Output and loading

The output contains the original processor/configuration files, a
`quantized_state_dict.pt`, and `quantization_manifest.json`. The floating-point
`model.safetensors` is deliberately not copied.

TorchAO weights cannot be loaded by the unmodified LeRobot
`PI05Policy.from_pretrained()`. Load them with:

```python
from light_weight import load_quantized_policy

policy = load_quantized_policy(
    "/path/to/pi05_int8wo",
    device="cuda",
    compile_mode="max-autotune-no-cudagraphs",
)
```

Only load `.pt` artifacts you created or trust. The loader must allow TorchAO
tensor subclasses during deserialization.

Quantization changes model numerics. Compare peak VRAM, chunk latency, action
error, and real-task success against the original checkpoint before allowing
unattended robot motion.

Deploy the finished artifact with the repository's dedicated entry point:

```bash
python ../deploy/deploy_pi05_light_weight.py \
  --interactive \
  --model-dir /path/to/pi05_int8wo \
  --tokenizer-dir /path/to/paligemma-tokenizer \
  --dry-run
```
