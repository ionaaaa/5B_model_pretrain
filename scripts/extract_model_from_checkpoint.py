#!/usr/bin/env python3
"""Extract a full-model state dict from a training checkpoint on CPU only."""

import argparse
import gc
import os
from pathlib import Path
import sys

import torch


# Full training checkpoints retain the Config dataclass, which pickle records as
# ``configs.Config``. When this script is executed as ``python scripts/...``,
# Python otherwise searches ``scripts/`` before the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


METADATA_KEYS = (
    "config",
    "shape_params",
    "epoch",
    "global_step",
    "train_loss",
    "val_loss",
    "best_val",
)


def format_size(num_bytes: int) -> str:
    """Return a compact GiB string for checkpoint size logging."""
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def extract_model_checkpoint(input_path: Path, output_path: Path, *, overwrite: bool, verify: bool):
    """Write a model-only checkpoint without instantiating the model or FSDP."""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if input_path == output_path:
        raise ValueError("Input and output checkpoint paths must be different.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input checkpoint does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}. "
            "Pass --overwrite to replace it."
        )

    # Training checkpoints contain trusted Python metadata (for example Config),
    # so they require weights_only=False with PyTorch 2.6 and newer.
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")

    model_state_dict = checkpoint["model_state_dict"]
    if not hasattr(model_state_dict, "keys"):
        raise ValueError("Checkpoint field 'model_state_dict' is not a state dict.")

    print(f"Input checkpoint: {input_path}")
    print(f"Checkpoint keys: {sorted(checkpoint.keys())}")
    print(f"Epoch: {checkpoint.get('epoch', 'not present')}")
    print(f"Global step: {checkpoint.get('global_step', 'not present')}")
    print(f"Has optimizer state: {'optimizer_state_dict' in checkpoint}")
    print(f"Model parameter tensors: {len(model_state_dict)}")

    # Reuse the loaded state dict directly: do not clone tensors or instantiate
    # the approximately 5B-parameter model.
    model_only_checkpoint = {
        "checkpoint_version": checkpoint.get("checkpoint_version", 1),
        "checkpoint_format": "model_only",
        "model_state_dict": model_state_dict,
    }
    for key in METADATA_KEYS:
        if key in checkpoint:
            model_only_checkpoint[key] = checkpoint[key]

    assert "optimizer_state_dict" not in model_only_checkpoint
    assert "scheduler_state_dict" not in model_only_checkpoint
    expected_model_keys = tuple(model_state_dict.keys())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_only_checkpoint, output_path)

    if not output_path.is_file():
        raise RuntimeError(f"Model-only checkpoint was not written: {output_path}")

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    reduction = 100.0 * (1.0 - output_size / input_size) if input_size else 0.0
    print(f"Input checkpoint size: {format_size(input_size)}")
    print(f"Output checkpoint size: {format_size(output_size)}")
    print(f"Reduction: {reduction:.1f}%")

    if verify:
        # Avoid retaining both full state dicts at once while reloading a large
        # output checkpoint solely for verification.
        del model_only_checkpoint
        del checkpoint
        gc.collect()
        converted = torch.load(output_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in converted
        assert "optimizer_state_dict" not in converted
        assert "scheduler_state_dict" not in converted
        assert tuple(converted["model_state_dict"].keys()) == expected_model_keys
        print("Verification: passed")

    print("Model-only checkpoint successfully saved")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract model parameters from a full FSDP training checkpoint."
    )
    parser.add_argument("--input", required=True, help="Path to the full checkpoint.")
    parser.add_argument("--output", required=True, help="Path for the model-only checkpoint.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output checkpoint.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Reload the output checkpoint and verify its structure and state-dict keys.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_model_checkpoint(
        Path(args.input),
        Path(args.output),
        overwrite=args.overwrite,
        verify=args.verify,
    )


if __name__ == "__main__":
    main()
