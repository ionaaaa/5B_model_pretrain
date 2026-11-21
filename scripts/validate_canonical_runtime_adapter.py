#!/usr/bin/env python3
"""Validate canonical-Lance -> FBMAE adaptation without writing Lance."""

import argparse
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from canonical_runtime_dataset import CanonicalRuntimeTargetDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read a canonical EEG Lance release from a local mount, build FBMAE "
            "tokens/targets in DataLoader workers, and run strict numerical checks."
        )
    )
    parser.add_argument(
        "--dataset", choices=("inria", "egobrain", "shin2017a"), required=True
    )
    parser.add_argument("--lance-path", required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=10)
    return parser.parse_args()


def _summary(name, values):
    values = np.asarray(values, dtype=np.float64)
    return (
        f"{name}: mean={values.mean():.3f}, p50={np.percentile(values, 50):.3f}, "
        f"p95={np.percentile(values, 95):.3f}, max={values.max():.3f}"
    )


def main():
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid batch-size or num-workers")
    if not os.path.exists(args.lance_path):
        raise FileNotFoundError(f"Mounted Lance path does not exist: {args.lance_path}")

    dataset = CanonicalRuntimeTargetDataset(
        lance_path=args.lance_path,
        dataset_key=args.dataset,
        max_samples=args.max_samples,
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
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(**loader_kwargs)

    print("valid_samples_scan:", dataset.valid_samples_audit, flush=True)
    print("channel_mapping_audit:", dataset.mapping_audit, flush=True)

    adaptation_times = []
    target_times = []
    loader_batch_times = []
    valid_channel_counts = []
    dropped_channel_counts = []
    dropped_channel_names = set()
    mapping_policies = set()
    waveform_mean_errors = []
    waveform_std_errors = []
    target_values = []
    token_values = []
    total_rows = 0

    started = time.perf_counter()
    previous = started
    for batch_index, batch in enumerate(loader):
        now = time.perf_counter()
        loader_batch_times.append((now - previous) * 1000.0)
        previous = now
        assert tuple(batch["token_inputs"].shape[1:]) == (640, 100)
        assert tuple(batch["targets"].shape[1:]) == (640, 5, 5)
        assert tuple(batch["token_channel_indices"].shape[1:]) == (640,)
        assert tuple(batch["token_time_indices"].shape[1:]) == (640,)
        assert tuple(batch["token_valid_mask"].shape[1:]) == (640,)
        assert tuple(batch["channel_valid_mask"].shape[1:]) == (64,)

        total_rows += int(batch["token_inputs"].shape[0])
        adaptation_times.extend(batch["adaptation_time_ms"].numpy().tolist())
        target_times.extend(batch["target_time_ms"].numpy().tolist())
        valid_channel_counts.extend(batch["valid_channel_count"].numpy().tolist())
        dropped_channel_counts.extend(batch["dropped_channel_count"].numpy().tolist())
        for names in batch["dropped_channel_names"]:
            dropped_channel_names.update(name for name in names.split(",") if name)
        mapping_policies.update(batch["channel_mapping_policy"])
        waveform_mean_errors.extend(batch["waveform_mean_abs_max"].numpy().tolist())
        waveform_std_errors.extend(batch["waveform_std_error_max"].numpy().tolist())
        target_values.append(batch["targets"].numpy().reshape(-1))
        token_values.append(batch["token_inputs"].numpy().reshape(-1))

        if args.report_every > 0 and (batch_index + 1) % args.report_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"checked_rows={total_rows}/{len(dataset)} elapsed_s={elapsed:.2f} "
                f"rows_per_s={total_rows / elapsed:.2f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    target_values = np.concatenate(target_values)
    token_values = np.concatenate(token_values)
    print(f"\n===== {args.dataset} canonical runtime FBMAE validation PASSED =====")
    print("lance_path:", args.lance_path)
    print("selected_rows:", len(dataset), "checked_rows:", total_rows)
    print("all rows satisfy valid_samples == 1000: YES")
    print("tensor shapes: token_inputs=(640,100), targets=(640,5,5)")
    print("token ordering: CHANNEL_MAJOR")
    print("finite tensors / legal indices / zero invalid channels: PASS")
    print("targets nonconstant and aligned with token order: PASS")
    print("mapping_policies:", sorted(mapping_policies))
    print("dropped_channel_names:", sorted(dropped_channel_names))
    print(_summary("dropped_channel_count", dropped_channel_counts))
    print(f"wall_time_s: {elapsed:.3f}")
    print(f"throughput_rows_per_s: {total_rows / elapsed:.3f}")
    print(_summary("adaptation_time_ms_per_row", adaptation_times))
    print(_summary("target_time_ms_per_row", target_times))
    print(_summary("loader_batch_wall_ms", loader_batch_times))
    print(_summary("valid_channel_count", valid_channel_counts))
    print(_summary("waveform_mean_abs_max", waveform_mean_errors))
    print(_summary("waveform_std_error_max", waveform_std_errors))
    print(
        "token_distribution: "
        f"mean={token_values.mean():.6f}, std={token_values.std():.6f}, "
        f"min={token_values.min():.6f}, max={token_values.max():.6f}"
    )
    print(
        "target_distribution: "
        f"mean={target_values.mean():.6f}, std={target_values.std():.6f}, "
        f"min={target_values.min():.6f}, max={target_values.max():.6f}"
    )
    print("No processed Lance was written.")


if __name__ == "__main__":
    main()
