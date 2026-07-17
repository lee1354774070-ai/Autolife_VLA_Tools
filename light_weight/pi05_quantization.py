#!/usr/bin/env python3
"""Shared TorchAO quantization and loading helpers for LeRobot PI0.5."""

from __future__ import annotations

import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn


ARTIFACT_NAME = "quantized_state_dict.pt"
MANIFEST_NAME = "quantization_manifest.json"
DEFAULT_EXCLUDES = (r"(?:^|\.)action_out_proj$",)
SUPPORTED_METHODS = ("int8wo", "int4wo")


@dataclass(frozen=True)
class LinearSelection:
    """Summary of the linear layers selected for quantization."""

    selected_names: tuple[str, ...]
    skipped_names: tuple[str, ...]
    selected_parameters: int
    total_linear_parameters: int


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def require_torchao() -> tuple[Callable[..., Any], type, type]:
    """Import the stable TorchAO APIs with a clear installation error."""

    try:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
    except ImportError as exc:
        raise RuntimeError(
            "TorchAO is required. Install a build compatible with the active "
            "PyTorch environment, for example: pip install torchao"
        ) from exc
    return quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig


def materialize_runtime_buffers(model: nn.Module, device: str) -> None:
    """Recreate PI0.5 buffers that safetensors intentionally does not save."""

    target = torch.device(device)
    for module in model.modules():
        position_ids = module._buffers.get("position_ids")
        if position_ids is not None and position_ids.device.type == "meta" and hasattr(module, "num_positions"):
            module.position_ids = torch.arange(module.num_positions, device=target).expand(1, -1)

        embed_scale = module._buffers.get("embed_scale")
        if embed_scale is not None and embed_scale.device.type == "meta" and hasattr(module, "scalar_embed_scale"):
            module.embed_scale = torch.tensor(
                module.scalar_embed_scale,
                dtype=embed_scale.dtype,
                device=target,
            )

        inv_freq = module._buffers.get("inv_freq")
        if inv_freq is not None and inv_freq.device.type == "meta" and hasattr(
            module, "compute_default_rope_parameters"
        ):
            restored, scaling = module.compute_default_rope_parameters(module.config, target)
            restored = restored.to(dtype=inv_freq.dtype)
            module.inv_freq = restored
            module.original_inv_freq = restored.clone()
            module.attention_scaling = scaling


def load_unquantized_policy(model_dir: Path, device: str):
    """Load PI0.5 without first allocating a second full floating-point model."""

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        raise RuntimeError("LeRobot with PI0.5 support is required.") from exc

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load the source checkpoint.") from exc

    config = PreTrainedConfig.from_pretrained(str(model_dir))
    config.compile_model = False
    # Build only model metadata first. Assigning the safetensors-backed state
    # avoids initializing a second 9+ GB PI0.5 parameter set before loading.
    config.device = "meta"
    with torch.device("meta"):
        policy = PI05Policy(config)

    state_dict = load_file(model_dir / "model.safetensors", device=device)
    policy.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    materialize_runtime_buffers(policy, device)
    policy.config.device = device
    policy.eval()
    policy.reset()
    return policy


def build_linear_filter(
    model: nn.Module,
    *,
    method: str,
    min_parameters: int,
    group_size: int,
    exclude_patterns: Iterable[str],
) -> tuple[Callable[[nn.Module, str], bool], LinearSelection]:
    """Select large compatible Linear layers and report every skipped layer."""

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported quantization method: {method}")
    if min_parameters < 0:
        raise ValueError("min_parameters must be non-negative")
    if method == "int4wo" and group_size not in (32, 64, 128, 256):
        raise ValueError("INT4 group_size must be one of: 32, 64, 128, 256")

    patterns = tuple(re.compile(pattern) for pattern in exclude_patterns)
    selected: list[str] = []
    skipped: list[str] = []
    selected_parameters = 0
    total_parameters = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        parameter_count = module.weight.numel()
        total_parameters += parameter_count
        excluded = any(pattern.search(name) for pattern in patterns)
        too_small = parameter_count < min_parameters
        incompatible_int4 = method == "int4wo" and module.in_features % group_size != 0
        if excluded or too_small or incompatible_int4:
            skipped.append(name)
            continue
        selected.append(name)
        selected_parameters += parameter_count

    selected_set = set(selected)

    def filter_fn(module: nn.Module, fqn: str) -> bool:
        return isinstance(module, nn.Linear) and fqn in selected_set

    summary = LinearSelection(
        selected_names=tuple(selected),
        skipped_names=tuple(skipped),
        selected_parameters=selected_parameters,
        total_linear_parameters=total_parameters,
    )
    return filter_fn, summary


def quantize_policy(
    policy: nn.Module,
    *,
    method: str,
    min_parameters: int,
    group_size: int,
    exclude_patterns: Iterable[str],
) -> LinearSelection:
    """Apply weight-only quantization in place to selected PI0.5 Linear layers."""

    quantize_, int8_config, int4_config = require_torchao()
    filter_fn, selection = build_linear_filter(
        policy,
        method=method,
        min_parameters=min_parameters,
        group_size=group_size,
        exclude_patterns=exclude_patterns,
    )
    if not selection.selected_names:
        raise RuntimeError("No compatible Linear layers were selected for quantization.")

    config = (
        int8_config(version=2)
        if method == "int8wo"
        else int4_config(group_size=group_size, version=2)
    )
    # Source weights were loaded directly on the requested device. Passing a
    # device here would make TorchAO call ``policy.to(device)`` and fail on
    # PI0.5's non-persistent meta buffers.
    quantize_(policy, config, filter_fn=filter_fn)
    return selection


def write_manifest(
    output_dir: Path,
    *,
    source_model: Path,
    method: str,
    device: str,
    group_size: int,
    min_parameters: int,
    exclude_patterns: Iterable[str],
    selection: LinearSelection,
    source_bytes: int,
    artifact_bytes: int,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format_version": 1,
        "policy_type": "pi05",
        "serialization": "torchao-state-dict",
        "artifact": ARTIFACT_NAME,
        "source_model": str(source_model.resolve()),
        "method": method,
        "device_used_for_quantization": device,
        "group_size": group_size if method == "int4wo" else None,
        "min_parameters": min_parameters,
        "exclude_patterns": list(exclude_patterns),
        "selected_module_count": len(selection.selected_names),
        "skipped_module_count": len(selection.skipped_names),
        "selected_parameters": selection.selected_parameters,
        "total_linear_parameters": selection.total_linear_parameters,
        "source_model_bytes": source_bytes,
        "quantized_artifact_bytes": artifact_bytes,
        "versions": {
            "python_torch": torch.__version__,
            "torchao": package_version("torchao"),
            "lerobot": package_version("lerobot"),
        },
        "selected_modules": list(selection.selected_names),
        "skipped_modules": list(selection.skipped_names),
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def read_manifest(model_dir: Path) -> dict[str, Any]:
    path = model_dir / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Quantization manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("policy_type") != "pi05":
        raise ValueError(f"Unsupported quantized artifact: {path}")
    return manifest


def load_quantized_policy(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    compile_mode: str | None = None,
    configure_runtime: Callable[[Any], None] | None = None,
):
    """Restore a TorchAO PI0.5 artifact without materializing FP weights.

    The artifact must be trusted because TorchAO tensor subclasses require
    ``torch.load(..., weights_only=False)``. Do not load files from unknown
    sources.
    """

    require_torchao()
    model_dir = Path(model_dir).expanduser().resolve()
    manifest = read_manifest(model_dir)
    artifact = model_dir / str(manifest.get("artifact", ARTIFACT_NAME))
    if not artifact.is_file():
        raise FileNotFoundError(f"Quantized state dict not found: {artifact}")

    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except ImportError as exc:
        raise RuntimeError("LeRobot with PI0.5 support is required.") from exc

    config = PreTrainedConfig.from_pretrained(str(model_dir))
    config.compile_model = False
    if configure_runtime is not None:
        configure_runtime(config)
    config.device = "meta"
    with torch.device("meta"):
        policy = PI05Policy(config)

    state_dict = torch.load(artifact, map_location=device, weights_only=False)
    policy.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    materialize_runtime_buffers(policy, device)

    meta_tensors = [
        name
        for name, tensor in list(policy.named_parameters()) + list(policy.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_tensors:
        raise RuntimeError("Artifact did not restore tensors: " + ", ".join(meta_tensors[:8]))

    policy.config.device = device
    policy.eval()
    policy.reset()
    if compile_mode:
        torch.set_float32_matmul_precision("high")
        policy.model.sample_actions = torch.compile(
            policy.model.sample_actions,
            mode=compile_mode,
        )
    return policy
