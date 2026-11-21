#!/usr/bin/env python3
"""Short-overfit diagnostic for runtime-adapted FBMAE data.

The script intentionally trains only a tiny ``EEGPretrainModel`` on 16--64
real samples.  It compares the correctly aligned targets with a deterministic
per-step cross-sample target derangement under identical initialization,
batches, masks, and optimizer settings.  Both models are evaluated against the
true aligned targets.  It never constructs the 5B model and never writes a
processed Lance dataset.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
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


SUPPORTED_DATASET = "inria"
TENSOR_FIELDS = (
    "token_inputs",
    "targets",
    "token_channel_indices",
    "token_time_indices",
    "token_valid_mask",
    "channel_valid_mask",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeat 16--64 real runtime-adapted samples through a tiny formal "
            "FBMAE model and compare aligned versus shuffled targets."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("inria", "egobrain", "shin2017a", "ds004408"),
        default="inria",
    )
    parser.add_argument("--lance-path", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help=(
            "Use 0.0 for the alignment sanity test. Production-like 0.8 masking "
            "is a separate stability test and may hide aligned/shuffled differences."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--min-loss-drop", type=float, default=0.10)
    parser.add_argument("--min-zero-baseline-advantage", type=float, default=0.10)
    parser.add_argument("--min-shuffle-advantage", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _reject_unready_dataset(dataset_name: str):
    if dataset_name == "ds004408":
        raise RuntimeError(
            "ds004408 is intentionally blocked: its electrode coordinate/channel "
            "mapping is not established, so it must not enter a model test."
        )
    if dataset_name in {"egobrain", "shin2017a"}:
        raise RuntimeError(
            f"{dataset_name} has no validated runtime channel/schema adapter in "
            "this checkout. Complete layer-1 numerical validation before running "
            "the short-overfit model test."
        )
    if dataset_name != SUPPORTED_DATASET:
        raise RuntimeError(f"Unsupported dataset: {dataset_name}")


def _load_real_samples(args) -> Dict[str, torch.Tensor]:
    dataset = InriaRuntimeTargetDataset(
        lance_path=args.lance_path,
        max_samples=args.samples,
        validate=True,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": min(args.batch_size, args.samples),
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
    pieces = {field: [] for field in TENSOR_FIELDS}
    for batch in DataLoader(**loader_kwargs):
        for field in TENSOR_FIELDS:
            pieces[field].append(batch[field].cpu())
    cached = {field: torch.cat(values, dim=0) for field, values in pieces.items()}
    if cached["token_inputs"].shape[0] != args.samples:
        raise AssertionError(
            f"loaded {cached['token_inputs'].shape[0]} samples, expected {args.samples}"
        )
    return cached


def _validate_cached_samples(data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    sample_count = int(data["token_inputs"].shape[0])
    expected_shapes = {
        "token_inputs": (sample_count, 640, 100),
        "targets": (sample_count, 640, 5, 5),
        "token_channel_indices": (sample_count, 640),
        "token_time_indices": (sample_count, 640),
        "token_valid_mask": (sample_count, 640),
        "channel_valid_mask": (sample_count, 64),
    }
    for field, expected in expected_shapes.items():
        actual = tuple(data[field].shape)
        if actual != expected:
            raise AssertionError(f"{field}.shape={actual}, expected={expected}")
        if not torch.isfinite(data[field]).all():
            raise AssertionError(f"{field} contains NaN or Inf")

    expected_channels = torch.arange(64).repeat_interleave(10).expand(sample_count, -1)
    expected_times = torch.arange(10).repeat(64).expand(sample_count, -1)
    if not torch.equal(data["token_channel_indices"], expected_channels):
        raise AssertionError("token_channel_indices are not channel-major")
    if not torch.equal(data["token_time_indices"], expected_times):
        raise AssertionError("token_time_indices are not channel-major")

    valid = data["token_valid_mask"] > 0.5
    channel_valid = data["channel_valid_mask"] > 0.5
    expected_token_valid = channel_valid.repeat_interleave(10, dim=1)
    if not torch.equal(valid, expected_token_valid):
        raise AssertionError("token_valid_mask does not match channel_valid_mask")
    valid_counts = valid.sum(dim=1)
    if int(valid_counts.min()) <= 0 or int(valid_counts.max()) > 640:
        raise AssertionError(f"invalid valid-token counts: {valid_counts.tolist()}")
    if not torch.all(valid_counts.remainder(10) == 0):
        raise AssertionError("valid-token count is not a whole number of channels")

    invalid_inputs = data["token_inputs"].masked_select(~valid[:, :, None].expand_as(data["token_inputs"]))
    invalid_targets = data["targets"].masked_select(~valid[:, :, None, None].expand_as(data["targets"]))
    if invalid_inputs.numel() and not torch.equal(invalid_inputs, torch.zeros_like(invalid_inputs)):
        raise AssertionError("invalid token_inputs are not zero")
    if invalid_targets.numel() and not torch.equal(invalid_targets, torch.zeros_like(invalid_targets)):
        raise AssertionError("invalid targets are not zero")

    valid_targets = data["targets"].masked_select(valid[:, :, None, None].expand_as(data["targets"]))
    target_std = float(valid_targets.float().std(unbiased=False))
    if not np.isfinite(target_std) or target_std <= 1e-6:
        raise AssertionError(f"valid targets are constant or invalid: std={target_std}")
    return {
        "valid_tokens_min": int(valid_counts.min()),
        "valid_tokens_mean": float(valid_counts.float().mean()),
        "valid_tokens_max": int(valid_counts.max()),
        "valid_channels_min": int(valid_counts.min()) // 10,
        "valid_channels_max": int(valid_counts.max()) // 10,
        "target_mean": float(valid_targets.float().mean()),
        "target_std": target_std,
        "target_min": float(valid_targets.min()),
        "target_max": float(valid_targets.max()),
    }


def _build_model(args, device: torch.device):
    torch.manual_seed(args.seed)
    return EEGPretrainModel(
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


def _build_schedule(sample_count: int, batch_size: int, steps: int, seed: int):
    rng = np.random.default_rng(seed)
    schedule: List[torch.Tensor] = []
    current: List[int] = []
    while len(schedule) < steps:
        current.extend(rng.permutation(sample_count).tolist())
        while len(current) >= batch_size and len(schedule) < steps:
            schedule.append(torch.tensor(current[:batch_size], dtype=torch.long))
            del current[:batch_size]
    return schedule


def _build_training_masks(schedule: Sequence[torch.Tensor], mask_ratio: float, seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return [
            build_batch_token_mask(
                batch_size=len(indices),
                num_tokens=640,
                mask_ratio=mask_ratio,
                device="cpu",
            ).cpu()
            for indices in schedule
        ]
    finally:
        np.random.set_state(state)


def _build_eval_masks(sample_count: int, batch_size: int, mask_ratio: float, seed: int):
    batches = [
        torch.arange(start, min(start + batch_size, sample_count), dtype=torch.long)
        for start in range(0, sample_count, batch_size)
    ]
    return batches, _build_training_masks(batches, mask_ratio, seed)


@torch.no_grad()
def _evaluate(
    model,
    criterion,
    data,
    target_map,
    batches,
    masks,
    device,
):
    model.eval()
    weighted_loss = 0.0
    total_valid = 0.0
    for indices, token_mask in zip(batches, masks):
        mapped = target_map[indices]
        inputs = data["token_inputs"][indices].to(device)
        targets = data["targets"][mapped].to(device)
        valid = (
            data["token_valid_mask"][indices]
            * data["token_valid_mask"][mapped]
        ).to(device)
        pred = model(
            token_inputs=apply_token_mask(inputs, token_mask.to(device)),
            token_channel_indices=data["token_channel_indices"][indices].to(device),
            token_time_indices=data["token_time_indices"][indices].to(device),
        )
        loss = criterion(pred=pred, target=targets, token_valid_mask=valid)
        if not torch.isfinite(loss) or not torch.isfinite(pred).all():
            raise FloatingPointError("non-finite prediction or evaluation loss")
        weight = float(valid.sum())
        weighted_loss += float(loss) * weight
        total_valid += weight
    if total_valid <= 0:
        raise AssertionError("evaluation has no valid target tokens")
    return weighted_loss / total_valid


def _train_condition(
    name,
    data,
    training_target_maps,
    evaluation_target_map,
    schedule,
    training_masks,
    eval_batches,
    eval_masks,
    args,
    device,
):
    model = _build_model(args, device)
    criterion = BalancedBandTokenTrajectoryLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = [
        (
            0,
            _evaluate(
                model,
                criterion,
                data,
                evaluation_target_map,
                eval_batches,
                eval_masks,
                device,
            ),
        )
    ]
    started = time.perf_counter()

    for step, (indices, token_mask, target_map) in enumerate(
        zip(schedule, training_masks, training_target_maps), start=1
    ):
        mapped = target_map[indices]
        inputs = data["token_inputs"][indices].to(device)
        targets = data["targets"][mapped].to(device)
        valid = (
            data["token_valid_mask"][indices]
            * data["token_valid_mask"][mapped]
        ).to(device)
        masked_inputs = apply_token_mask(inputs, token_mask.to(device))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=data["token_channel_indices"][indices].to(device),
            token_time_indices=data["token_time_indices"][indices].to(device),
        )
        loss = criterion(pred=pred, target=targets, token_valid_mask=valid)
        if not torch.isfinite(loss) or not torch.isfinite(pred).all():
            raise FloatingPointError(f"{name} produced non-finite values at step {step}")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError(f"{name} produced non-finite gradients at step {step}")
        optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError(f"{name} produced non-finite parameters at step {step}")

        if step % args.eval_every == 0 or step == args.steps:
            eval_loss = _evaluate(
                model,
                criterion,
                data,
                evaluation_target_map,
                eval_batches,
                eval_masks,
                device,
            )
            history.append((step, eval_loss))
            print(
                f"condition={name} step={step:04d}/{args.steps} "
                f"train_loss={loss.detach().cpu().item():.6f} "
                f"fixed_eval_loss={eval_loss:.6f}",
                flush=True,
            )

    initial_loss = history[0][1]
    final_loss = history[-1][1]
    relative_drop = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
    return {
        "name": name,
        "history": history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_drop": relative_drop,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_short_overfit(data, args):
    stats = _validate_cached_samples(data)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    sample_count = int(data["token_inputs"].shape[0])
    schedule = _build_schedule(sample_count, args.batch_size, args.steps, args.seed + 1)
    training_masks = _build_training_masks(schedule, args.mask_ratio, args.seed + 2)
    eval_batches, eval_masks = _build_eval_masks(
        sample_count, args.batch_size, args.mask_ratio, args.seed + 3
    )
    aligned_map = torch.arange(sample_count, dtype=torch.long)
    aligned_training_maps = [aligned_map] * args.steps
    shuffle_rng = np.random.default_rng(args.seed + 4)
    shuffled_training_maps = [
        torch.roll(
            aligned_map,
            shifts=int(shuffle_rng.integers(1, sample_count)),
        )
        for _ in range(args.steps)
    ]
    if any(torch.any(target_map == aligned_map) for target_map in shuffled_training_maps):
        raise AssertionError("a shuffled training target map is not a derangement")

    aligned = _train_condition(
        "aligned", data, aligned_training_maps, aligned_map, schedule, training_masks,
        eval_batches, eval_masks, args, device,
    )
    shuffled = _train_condition(
        "shuffled", data, shuffled_training_maps, aligned_map, schedule, training_masks,
        eval_batches, eval_masks, args, device,
    )
    criterion = BalancedBandTokenTrajectoryLoss()
    with torch.no_grad():
        zero_baseline_loss = float(
            criterion(
                pred=torch.zeros_like(data["targets"]),
                target=data["targets"],
                token_valid_mask=data["token_valid_mask"],
            )
        )
    zero_baseline_advantage = (
        zero_baseline_loss - aligned["final_loss"]
    ) / max(abs(zero_baseline_loss), 1e-12)
    shuffle_advantage = (
        shuffled["final_loss"] - aligned["final_loss"]
    ) / max(abs(shuffled["final_loss"]), 1e-12)
    aligned_drop_pass = aligned["relative_drop"] >= args.min_loss_drop
    zero_baseline_pass = zero_baseline_advantage >= args.min_zero_baseline_advantage
    shuffle_pass = shuffle_advantage >= args.min_shuffle_advantage
    return {
        "data_stats": stats,
        "aligned": aligned,
        "shuffled": shuffled,
        "zero_baseline_loss": zero_baseline_loss,
        "zero_baseline_advantage": zero_baseline_advantage,
        "shuffle_advantage": shuffle_advantage,
        "aligned_drop_pass": aligned_drop_pass,
        "zero_baseline_pass": zero_baseline_pass,
        "shuffle_pass": shuffle_pass,
    }


def main():
    args = parse_args()
    _reject_unready_dataset(args.dataset)
    if not os.path.exists(args.lance_path):
        raise FileNotFoundError(f"Mounted Lance path does not exist: {args.lance_path}")
    if not 16 <= args.samples <= 64:
        raise ValueError("--samples must be in [16, 64]")
    if args.steps < 20:
        raise ValueError("--steps must be at least 20 for a short-overfit test")
    if args.batch_size <= 0 or args.batch_size > args.samples:
        raise ValueError("--batch-size must be positive and no greater than --samples")
    if args.num_workers < 0 or args.eval_every <= 0:
        raise ValueError("invalid worker/evaluation settings")
    if args.d_model <= 0 or args.n_heads <= 0 or args.depth <= 0:
        raise ValueError("tiny model dimensions must be positive")
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads")
    if not 0.0 <= args.mask_ratio < 1.0:
        raise ValueError("--mask-ratio must satisfy 0 <= ratio < 1")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.mask_ratio > 0.5:
        print(
            "WARNING: mask_ratio > 0.5 is useful for production-like stability, "
            "but is too strong for a decisive short alignment-overfit test. "
            "Use --mask-ratio 0.0 for the alignment diagnostic.",
            flush=True,
        )

    data = _load_real_samples(args)
    result = run_short_overfit(data, args)
    stats = result["data_stats"]
    aligned = result["aligned"]
    shuffled = result["shuffled"]

    print("\n===== Runtime short-overfit diagnostic =====")
    print("dataset:", args.dataset)
    print("samples:", args.samples, "steps_per_condition:", args.steps)
    print("device:", args.device, "tiny_model_only: YES")
    print("negative_control: per-step target derangement; evaluation target=aligned")
    print(
        "valid_tokens_per_sample:",
        f"min={stats['valid_tokens_min']} mean={stats['valid_tokens_mean']:.2f} "
        f"max={stats['valid_tokens_max']}",
    )
    print(
        "valid_channels_per_sample:",
        f"min={stats['valid_channels_min']} max={stats['valid_channels_max']}",
    )
    print(
        "valid_target_distribution:",
        f"mean={stats['target_mean']:.6f} std={stats['target_std']:.6f} "
        f"min={stats['target_min']:.6f} max={stats['target_max']:.6f}",
    )
    print(
        "aligned:",
        f"initial={aligned['initial_loss']:.6f} final={aligned['final_loss']:.6f} "
        f"drop={100.0 * aligned['relative_drop']:.2f}%",
    )
    print(
        "shuffled:",
        f"initial={shuffled['initial_loss']:.6f} final={shuffled['final_loss']:.6f} "
        f"drop={100.0 * shuffled['relative_drop']:.2f}%",
    )
    print("zero_predictor_loss:", f"{result['zero_baseline_loss']:.6f}")
    print(
        "aligned_advantage_over_zero_predictor:",
        f"{100.0 * result['zero_baseline_advantage']:.2f}%",
    )
    print("aligned_advantage_over_shuffled:", f"{100.0 * result['shuffle_advantage']:.2f}%")
    print("finite predictions/loss/gradients/parameters: PASS")
    print("aligned_loss_drop:", "PASS" if result["aligned_drop_pass"] else "FAIL")
    print("aligned_beats_zero_predictor:", "PASS" if result["zero_baseline_pass"] else "FAIL")
    print("aligned_better_than_shuffled:", "PASS" if result["shuffle_pass"] else "FAIL")
    print("No 5B model was constructed. No processed Lance was written.")

    failures = []
    if not result["aligned_drop_pass"]:
        failures.append(
            f"aligned loss drop {aligned['relative_drop']:.4f} < {args.min_loss_drop:.4f}"
        )
    if not result["zero_baseline_pass"]:
        failures.append(
            f"aligned advantage over zero predictor "
            f"{result['zero_baseline_advantage']:.4f} < "
            f"{args.min_zero_baseline_advantage:.4f}"
        )
    if not result["shuffle_pass"]:
        failures.append(
            f"aligned advantage {result['shuffle_advantage']:.4f} <= "
            f"{args.min_shuffle_advantage:.4f}"
        )
    if failures:
        raise AssertionError("Short-overfit diagnostic failed: " + "; ".join(failures))
    print("===== Runtime short-overfit validation PASSED =====")


if __name__ == "__main__":
    main()
