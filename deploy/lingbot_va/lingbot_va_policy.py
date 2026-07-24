"""Model-only LingBot-VA loading helpers shared by local and remote deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_checkpoint_action_bounds(model_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the physical 16-D action range saved with a LingBot checkpoint."""

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("Reading LingBot action bounds requires safetensors.") from exc

    model_dir = Path(model_dir).expanduser().resolve()
    candidates = sorted(
        model_dir.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors")
    )
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one checkpoint postprocessor stats file, got "
            f"{[str(path) for path in candidates]}."
        )
    tensors = load_file(candidates[0])
    try:
        lower = np.asarray(tensors["action.min"], dtype=np.float32).reshape(-1)
        upper = np.asarray(tensors["action.max"], dtype=np.float32).reshape(-1)
    except KeyError as exc:
        raise ValueError(f"Checkpoint action statistics are missing {exc.args[0]!r}.") from exc
    if (
        lower.shape != (16,)
        or upper.shape != (16,)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.all(lower <= upper)
    ):
        raise ValueError(
            f"Invalid 16-D checkpoint action bounds: min={lower.shape}, max={upper.shape}."
        )
    return lower, upper


def unwrap_lingbot_policy(policy: Any) -> Any:
    """Return the adapter-injected LingBot policy underneath an optional PEFT wrapper."""

    get_base_model = getattr(policy, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else policy


def load_lingbot_policy(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    base_model: str | Path | None = None,
    wan_model: str | Path = "robbyant/lingbot-va-base",
    text_encoder_device: str = "cpu",
    attn_mode: str = "torch",
    action_inference_steps: int = 50,
    video_inference_steps: int = 20,
    guidance_scale: float = 5.0,
    offline: bool = False,
):
    """Load a LingBot base policy, its LoRA adapter, and saved I/O processors."""

    try:
        from peft import PeftConfig, PeftModel

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
    except ImportError as exc:
        raise RuntimeError(
            "LingBot-VA deployment requires LeRobot 0.6 and PEFT. "
            "Install PEFT with: pip install 'peft>=0.18,<1'."
        ) from exc

    model_dir = Path(model_dir).expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(str(model_dir))
    if config.type != "lingbot_va":
        raise ValueError(f"Expected a lingbot_va checkpoint, got {config.type!r}.")
    if len(config.used_action_channel_ids) != 16:
        raise ValueError(
            "This deployment expects the trained 16-D base action layout; "
            f"checkpoint channels are {config.used_action_channel_ids}."
        )

    config.device = device
    config.dtype = "bfloat16"
    config.attn_mode = attn_mode
    config.text_encoder_device = text_encoder_device
    config.wan_pretrained_path = str(wan_model)
    config.action_num_inference_steps = action_inference_steps
    config.num_inference_steps = video_inference_steps
    config.guidance_scale = guidance_scale

    adapter_config = PeftConfig.from_pretrained(str(model_dir))
    base_model = base_model or adapter_config.base_model_name_or_path
    if not base_model:
        raise ValueError("adapter_config.json does not identify a base LingBot model.")

    base_policy = LingBotVAPolicy.from_pretrained(
        str(base_model),
        config=config,
        local_files_only=offline,
    )
    policy = PeftModel.from_pretrained(
        base_policy,
        str(model_dir),
        config=adapter_config,
        is_trainable=False,
    )
    policy = policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    if hasattr(policy, "reset"):
        policy.reset()
    return policy, preprocessor, postprocessor


def action_tensor(value: Any):
    """Extract an action tensor from processor output without changing its shape."""

    import torch

    if isinstance(value, dict):
        value = value.get("action")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value
