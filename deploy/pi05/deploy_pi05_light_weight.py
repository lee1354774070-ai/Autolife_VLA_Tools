#!/usr/bin/env python3
"""Run a TorchAO weight-only LeRobot PI0.5 policy on an AutoLife robot.

This entry point reuses deploy_pi05.py for camera synchronization, policy
processing, robot schema conversion, ROS2 publishing, and interactive session
control. Only the checkpoint loader is replaced.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = SCRIPT_DIR.parent
REPO_DIR = DEPLOY_DIR.parent
for import_dir in (SCRIPT_DIR, DEPLOY_DIR, REPO_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from pi05 import deploy_pi05 as base  # noqa: E402
from light_weight.pi05_quantization import (  # noqa: E402
    load_quantized_policy,
    read_manifest,
)


def release_version(value: str) -> tuple[int, ...]:
    """Return the numeric release tuple while ignoring CUDA/local suffixes."""

    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"{package} is required by this quantized checkpoint. Install its compatible version."
        ) from exc


def validate_quantized_environment(model_dir: Path) -> dict:
    """Reject incompatible tensor-subclass serialization before robot startup."""

    manifest = read_manifest(model_dir)
    saved = manifest.get("versions", {})
    current = {
        "python_torch": installed_version("torch"),
        "torchao": installed_version("torchao"),
        "lerobot": installed_version("lerobot"),
    }
    mismatches: list[str] = []
    for name in ("python_torch", "torchao", "lerobot"):
        expected = str(saved.get(name, ""))
        actual = current[name]
        if not expected:
            continue
        expected_release = release_version(expected)
        actual_release = release_version(actual)
        compare_parts = 2 if name in {"python_torch", "lerobot"} else 3
        if expected_release[:compare_parts] != actual_release[:compare_parts]:
            mismatches.append(f"{name}: artifact={expected}, installed={actual}")
    if mismatches:
        raise RuntimeError(
            "Quantized checkpoint version mismatch; re-quantize in this deployment environment "
            "or install matching packages: " + "; ".join(mismatches)
        )
    return manifest


def load_light_weight_policy(
    model_dir: Path,
    device: str,
    tokenizer_dir: Path | None = None,
    compile_mode: str = "max-autotune-no-cudagraphs",
    n_action_steps: int | None = None,
    rtc_enabled: bool = False,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
):
    """Load the quantized policy plus the original LeRobot processors."""

    manifest = validate_quantized_environment(model_dir)
    effective_compile_mode = None if compile_mode in {"checkpoint", "disabled"} else compile_mode
    print(
        f"quantized policy: method={manifest.get('method')}, "
        f"artifact={manifest.get('artifact')}, compile={effective_compile_mode or 'disabled'}"
    )
    policy = load_quantized_policy(
        model_dir,
        device=device,
        compile_mode=effective_compile_mode,
        configure_runtime=lambda config: base.configure_policy_runtime(
            config,
            n_action_steps,
            rtc_enabled,
            rtc_execution_horizon,
            rtc_max_guidance_weight,
        ),
    )

    try:
        from lerobot.policies import make_pre_post_processors
    except ImportError as exc:
        raise RuntimeError("LeRobot policy processors are unavailable.") from exc

    overrides: dict[str, dict[str, str]] = {}
    if tokenizer_dir is not None:
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}")
        overrides = {"tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)}}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(model_dir),
        preprocessor_overrides=overrides,
    )
    return policy, preprocessor, postprocessor


def main() -> None:
    # INT8/INT4 tensor subclasses generally benefit from torch.compile, while
    # CUDA graphs caused excessive peak memory on the 16 GB deployment GPU.
    os.environ.setdefault("PI05_COMPILE_MODE", "max-autotune-no-cudagraphs")
    os.environ.setdefault("PI05_REQUIRE_MODEL_DIR", "1")
    base.__doc__ = __doc__
    base.load_policy = load_light_weight_policy
    base.main()


if __name__ == "__main__":
    main()
