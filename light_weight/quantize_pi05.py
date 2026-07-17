#!/usr/bin/env python3
"""Create a loadable TorchAO weight-only PI0.5 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch

if __package__:
    from .pi05_quantization import (
        ARTIFACT_NAME,
        DEFAULT_EXCLUDES,
        MANIFEST_NAME,
        SUPPORTED_METHODS,
        build_linear_filter,
        load_quantized_policy,
        load_unquantized_policy,
        quantize_policy,
        write_manifest,
    )
else:  # Direct execution: python light_weight/quantize_pi05.py
    from pi05_quantization import (
        ARTIFACT_NAME,
        DEFAULT_EXCLUDES,
        MANIFEST_NAME,
        SUPPORTED_METHODS,
        build_linear_filter,
        load_quantized_policy,
        load_unquantized_policy,
        quantize_policy,
        write_manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize LeRobot PI0.5 Linear weights with TorchAO. The source "
            "checkpoint is never modified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, required=True, help="Source LeRobot checkpoint")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New quantized checkpoint; required unless --dry-run is used",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default="int8wo",
        help="INT8 or experimental INT4 weight-only quantization",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used while loading and quantizing",
    )
    parser.add_argument(
        "--min-parameters",
        type=int,
        default=16_384,
        help="Skip Linear weights smaller than this number of elements",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=(32, 64, 128, 256),
        default=128,
        help="INT4 quantization group size; ignored by INT8",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="REGEX",
        help="Additional fully-qualified module-name regex to leave unquantized",
    )
    parser.add_argument(
        "--quantize-action-output",
        action="store_true",
        help="Also quantize action_out_proj; disabled by default for action accuracy",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load and list selection without quantizing")
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="Print selected and skipped Linear module names",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Release the source policy and reload the finished artifact",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser.parse_args()


def validate_paths(
    source: Path,
    output: Path | None,
    *,
    dry_run: bool,
    overwrite: bool,
) -> tuple[Path, Path | None]:
    source = source.expanduser().resolve()
    if not (source / "config.json").is_file() or not (source / "model.safetensors").is_file():
        raise FileNotFoundError(f"Not a local LeRobot checkpoint: {source}")
    if output is None:
        if dry_run:
            return source, None
        raise ValueError("--output-dir is required unless --dry-run is used")
    output = output.expanduser().resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError(
            "output-dir must be separate from the source checkpoint and cannot be its parent"
        )
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists; use --overwrite: {output}")
        shutil.rmtree(output)
    return source, output


def copy_checkpoint_metadata(source: Path, output: Path) -> None:
    """Copy processors/configuration while deliberately excluding FP weights."""

    ignored = {"model.safetensors", ARTIFACT_NAME, MANIFEST_NAME}
    shutil.copytree(source, output, ignore=lambda _path, names: [name for name in names if name in ignored])
    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["compile_model"] = False
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def print_selection(selection, list_modules: bool) -> None:
    total = selection.total_linear_parameters
    ratio = 100.0 * selection.selected_parameters / total if total else 0.0
    print(f"Selected Linear modules : {len(selection.selected_names)}")
    print(f"Skipped Linear modules  : {len(selection.skipped_names)}")
    print(f"Selected parameters      : {selection.selected_parameters:,} / {total:,} ({ratio:.1f}%)")
    if list_modules:
        print("Selected modules:")
        for name in selection.selected_names:
            print(f"  + {name}")
        print("Skipped modules:")
        for name in selection.skipped_names:
            print(f"  - {name}")


def main() -> int:
    args = parse_args()
    source, output = validate_paths(
        args.model_dir,
        args.output_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    excludes = list(args.exclude)
    if not args.quantize_action_output:
        excludes.extend(DEFAULT_EXCLUDES)

    print(f"Loading source policy: {source}")
    policy = load_unquantized_policy(source, args.device)
    filter_fn, selection = build_linear_filter(
        policy,
        method=args.method,
        min_parameters=args.min_parameters,
        group_size=args.group_size,
        exclude_patterns=excludes,
    )
    del filter_fn
    print_selection(selection, args.list_modules)
    if args.dry_run:
        print("Dry run complete; no output was written.")
        return 0

    assert output is not None
    print(f"Applying {args.method} quantization...")
    selection = quantize_policy(
        policy,
        method=args.method,
        min_parameters=args.min_parameters,
        group_size=args.group_size,
        exclude_patterns=excludes,
    )
    copy_checkpoint_metadata(source, output)

    artifact = output / ARTIFACT_NAME
    torch.save(policy.state_dict(), artifact)
    source_weights = source / "model.safetensors"
    manifest = write_manifest(
        output,
        source_model=source,
        method=args.method,
        device=args.device,
        group_size=args.group_size,
        min_parameters=args.min_parameters,
        exclude_patterns=excludes,
        selection=selection,
        source_bytes=source_weights.stat().st_size,
        artifact_bytes=artifact.stat().st_size,
    )

    source_gib = manifest["source_model_bytes"] / 1024**3
    output_gib = manifest["quantized_artifact_bytes"] / 1024**3
    print(f"Source weight file        : {source_gib:.2f} GiB")
    print(f"Quantized artifact        : {output_gib:.2f} GiB")
    print(f"Saved quantized checkpoint: {output}")

    if args.verify_load:
        print("Releasing source policy and verifying artifact reload...")
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restored = load_quantized_policy(output, device=args.device)
        restored_count = sum(1 for _ in restored.parameters())
        print(f"Artifact reload succeeded ({restored_count} parameter tensors).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
