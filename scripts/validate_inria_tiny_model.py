#!/usr/bin/env python3
"""Run one real Inria runtime batch through the formal FBMAE model path.

This is deliberately a tiny CPU smoke test.  It uses the production model,
masking, and loss implementations, but never constructs the 5B configuration
and never writes a processed Lance dataset.
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inria_runtime_dataset import InriaRuntimeTargetDataset  # noqa: E402
from losses import BalancedBandTokenTrajectoryLoss  # noqa: E402
from masking import apply_token_mask  # noqa: E402
from model import EEGPretrainModel  # noqa: E402
from trainer import build_batch_token_mask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read one real mounted Inria batch, dynamically build FBMAE targets, "
            "and run a tiny formal-model forward/backward/optimizer step."
        )
    )
    parser.add_argument("--lance-path", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is sufficient and is the default; this never builds the 5B model.",
    )
    return parser.parse_args()


def _assert_batch_schema(batch):
    batch_size = int(batch["token_inputs"].shape[0])
    expected = {
        "token_inputs": (batch_size, 640, 100),
        "targets": (batch_size, 640, 5, 5),
        "token_channel_indices": (batch_size, 640),
        "token_time_indices": (batch_size, 640),
        "token_valid_mask": (batch_size, 640),
        "channel_valid_mask": (batch_size, 64),
    }
    for name, shape in expected.items():
        actual = tuple(batch[name].shape)
        if actual != shape:
            raise AssertionError(f"{name}.shape={actual}, expected={shape}")
        if not torch.isfinite(batch[name]).all():
            raise AssertionError(f"{name} contains NaN or Inf")

    expected_channels = torch.arange(64).repeat_interleave(10)
    expected_times = torch.arange(10).repeat(64)
    if not torch.equal(batch["token_channel_indices"].cpu(), expected_channels.expand(batch_size, -1)):
        raise AssertionError("token_channel_indices are not channel-major")
    if not torch.equal(batch["token_time_indices"].cpu(), expected_times.expand(batch_size, -1)):
        raise AssertionError("token_time_indices are not channel-major")


def run_tiny_step(batch, args):
    """Run the current masking/model/loss path and return audit metrics."""
    _assert_batch_schema(batch)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    torch.manual_seed(args.seed)
    model = EEGPretrainModel(
        input_dim=100,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        mlp_ratio=4.0,
        dropout=0.0,
        n_channels=64,
        n_time_patches=10,
        n_bands=5,
        frames_per_patch=5,
    ).to(device)
    criterion = BalancedBandTokenTrajectoryLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    token_inputs = batch["token_inputs"].to(device)
    targets = batch["targets"].to(device)
    channel_indices = batch["token_channel_indices"].to(device)
    time_indices = batch["token_time_indices"].to(device)
    token_valid_mask = batch["token_valid_mask"].to(device)

    # The Dataset has already generated targets from the unmasked waveform.
    # Keep an exact snapshot and prove that masking touches token_inputs only.
    targets_before_mask = targets.clone()
    token_mask = build_batch_token_mask(
        batch_size=token_inputs.shape[0],
        num_tokens=token_inputs.shape[1],
        mask_ratio=args.mask_ratio,
        device=device,
    )
    masked_inputs = apply_token_mask(token_inputs, token_mask)
    if not torch.equal(targets, targets_before_mask):
        raise AssertionError("targets changed while applying the token mask")
    if not torch.equal(masked_inputs[token_mask.bool()], torch.zeros_like(masked_inputs[token_mask.bool()])):
        raise AssertionError("masked token_inputs are not exactly zero")

    parameter_snapshots = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    pred = model(
        token_inputs=masked_inputs,
        token_channel_indices=channel_indices,
        token_time_indices=time_indices,
    )
    forward_seconds = time.perf_counter() - started

    expected_pred_shape = (token_inputs.shape[0], 640, 5, 5)
    if tuple(pred.shape) != expected_pred_shape:
        raise AssertionError(f"pred.shape={tuple(pred.shape)}, expected={expected_pred_shape}")
    if not torch.isfinite(pred).all():
        raise AssertionError("prediction contains NaN or Inf")

    loss = criterion(
        pred=pred,
        target=targets,
        token_valid_mask=token_valid_mask,
    )
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"loss is not a finite scalar: {loss}")

    started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients:
        raise AssertionError("backward produced no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("backward produced a NaN or Inf gradient")
    gradient_norm = torch.sqrt(sum(gradient.detach().float().pow(2).sum() for gradient in gradients))
    if not torch.isfinite(gradient_norm):
        raise AssertionError("global gradient norm is not finite")

    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise AssertionError("optimizer step produced a NaN or Inf parameter")
    changed_parameter_tensors = sum(
        not torch.equal(parameter.detach(), parameter_snapshots[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if changed_parameter_tensors == 0:
        raise AssertionError("optimizer step did not update any parameter tensor")

    return {
        "pred_shape": tuple(pred.shape),
        "loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "gradient_tensors": len(gradients),
        "changed_parameter_tensors": changed_parameter_tensors,
        "masked_tokens_per_sample": token_mask.sum(dim=1).detach().cpu().tolist(),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def main():
    args = parse_args()
    if not os.path.exists(args.lance_path):
        raise FileNotFoundError(f"Mounted Inria Lance path does not exist: {args.lance_path}")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid --batch-size or --num-workers")
    if args.d_model <= 0 or args.n_heads <= 0 or args.depth <= 0:
        raise ValueError("tiny model dimensions must be positive")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads")
    if not 0.0 <= args.mask_ratio < 1.0:
        raise ValueError("--mask-ratio must satisfy 0 <= ratio < 1")

    dataset = InriaRuntimeTargetDataset(
        lance_path=args.lance_path,
        max_samples=args.batch_size,
        validate=True,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )

    print("valid_samples_scan:", dataset.valid_samples_audit, flush=True)
    batch = next(iter(DataLoader(**loader_kwargs)))
    metrics = run_tiny_step(batch, args)

    print("\n===== Inria tiny formal-model validation PASSED =====")
    print("device:", args.device)
    print("tiny_model: EEGPretrainModel from model.py")
    print(
        "tiny_config:",
        {
            "input_dim": 100,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "depth": args.depth,
            "n_channels": 64,
            "n_time_patches": 10,
            "n_bands": 5,
            "frames_per_patch": 5,
        },
    )
    print("model_parameters:", metrics["model_parameters"])
    print("real_batch_shapes: token_inputs=(B,640,100), targets=(B,640,5,5)")
    print("targets generated before masking and unchanged by masking: PASS")
    print("token ordering: CHANNEL_MAJOR")
    print("masked_tokens_per_sample:", metrics["masked_tokens_per_sample"])
    print("pred.shape:", metrics["pred_shape"])
    print("loss:", f"{metrics['loss']:.8f}", "finite=YES")
    print("backward: PASS")
    print("gradient_tensors:", metrics["gradient_tensors"])
    print("gradient_norm:", f"{metrics['gradient_norm']:.8f}", "finite=YES")
    print("optimizer_step: PASS")
    print("changed_parameter_tensors:", metrics["changed_parameter_tensors"])
    print("forward_seconds:", f"{metrics['forward_seconds']:.4f}")
    print("backward_seconds:", f"{metrics['backward_seconds']:.4f}")
    print("No 5B model was constructed. No processed Lance was written.")


if __name__ == "__main__":
    main()
