#!/usr/bin/env python3
"""Analyze constant-valued suffixes in processed EEG Lance shards.

The processed dataset stores a 10-second signal as flattened, channel-major
``token_inputs`` patches.  This script reconstructs the time axis, excludes
invalid and whole-recording-constant channels, and defines a row's tail as the
constant suffix shared by every remaining channel (the minimum channel tail).

Examples
--------
Local smoke test (the sample download has one redundant directory level)::

    python analyze_constant_tails.py \
      --input dataset/processed_parts/part_00000000_00010000.lance \
      --limit-rows 200 --output-prefix reports/smoke_constant_tail

Several shards mounted from one COS bucket::

    python analyze_constant_tails.py \
      --input \
        cos://jianyu-1442740494/AAAI2026/home/jianyu/AAAI2026/processed_parts/part_00090000_00100000.lance \
        cos://jianyu-1442740494/AAAI2026/home/jianyu/AAAI2026/processed_parts/part_00060000_00070000.lance \
      --cos-mount-root /mnt/jianyu-1442740494 \
      --dataset-name AAAI2026 \
      --output-prefix reports/aaai2026_constant_tail

``--cos-mount-root`` is the directory at which the named bucket itself is
mounted, so the bucket component is omitted by default.  Add
``--cos-include-bucket`` when the mount root contains one directory per bucket.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import lance
import numpy as np
import pyarrow as pa


DEFAULT_THRESHOLDS_SECONDS = (0.5, 1.0, 2.0)
WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find common constant tails in processed EEG Lance rows and report "
            "row-weighted distributions by dataset, shard, and subject."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        nargs="+",
        action="extend",
        default=[],
        help="One or more local/mounted Lance paths or cos:// URIs. May be repeated.",
    )
    parser.add_argument(
        "--input-list",
        action="append",
        default=[],
        help="Text file containing one input path per line; blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--cos-mount-root",
        default=None,
        help="Local directory where a COS bucket is mounted; required for cos:// inputs.",
    )
    parser.add_argument(
        "--cos-include-bucket",
        action="store_true",
        help="Map cos://bucket/key to MOUNT/bucket/key instead of MOUNT/key.",
    )
    parser.add_argument(
        "--dataset-name",
        action="append",
        default=[],
        help=(
            "Override dataset labels. Supply once for all inputs or once per input. "
            "Otherwise a non-empty dataset_key column is used."
        ),
    )
    parser.add_argument("--signal-column", default="token_inputs")
    parser.add_argument("--dataset-column", default="dataset_key")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--valid-mask-column", default="channel_valid_mask")
    parser.add_argument("--sampling-rate", type=float, default=100.0, help="Samples per second.")
    parser.add_argument(
        "--patch-stride-samples",
        type=int,
        default=None,
        help="Patch stride used during preprocessing; default assumes stride == patch_len.",
    )
    parser.add_argument(
        "--min-tail-seconds",
        type=float,
        default=0.1,
        help="Minimum shared suffix length counted as a constant tail.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for equality to a channel's final value.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.0,
        help="Relative tolerance for equality to a channel's final value.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Read at most this many rows from each input (useful for smoke tests).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress after this many rows per shard; 0 disables progress.",
    )
    parser.add_argument(
        "--output-prefix",
        default="constant_tail_report",
        help="Writes PREFIX.summary.csv and PREFIX.report.json.",
    )
    parser.add_argument(
        "--row-details",
        default=None,
        help="Optional CSV path for per-row tail lengths and diagnostics.",
    )
    args = parser.parse_args(argv)

    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be > 0")
    if args.min_tail_seconds < 0:
        parser.error("--min-tail-seconds must be >= 0")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.limit_rows is not None and args.limit_rows <= 0:
        parser.error("--limit-rows must be > 0")
    if args.patch_stride_samples is not None and args.patch_stride_samples <= 0:
        parser.error("--patch-stride-samples must be > 0")
    return args


def load_inputs(cli_inputs: Sequence[str], list_files: Sequence[str]) -> List[str]:
    inputs = list(cli_inputs)
    for list_file in list_files:
        with open(list_file, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    inputs.append(line)
    if not inputs:
        raise ValueError("No inputs supplied. Use --input and/or --input-list.")
    return inputs


def map_input_path(
    raw: str, cos_mount_root: Optional[str], cos_include_bucket: bool
) -> Path:
    if WINDOWS_DRIVE_PATH.match(raw):
        return find_lance_root(Path(raw).expanduser())
    parsed = urlparse(raw)
    if parsed.scheme.lower() == "cos":
        if not cos_mount_root:
            raise ValueError(
                f"COS URI requires --cos-mount-root because data are read through the mount: {raw}"
            )
        relative = PurePosixPath(parsed.path.lstrip("/"))
        mapped = Path(cos_mount_root)
        if cos_include_bucket:
            mapped /= parsed.netloc
        mapped = mapped.joinpath(*relative.parts)
    elif parsed.scheme and parsed.scheme.lower() != "file":
        raise ValueError(
            f"Unsupported URI scheme {parsed.scheme!r} for {raw!r}; pass the mounted local path "
            "or use cos:// together with --cos-mount-root."
        )
    else:
        mapped = Path(parsed.path if parsed.scheme.lower() == "file" else raw).expanduser()

    return find_lance_root(mapped)


def find_lance_root(path: Path) -> Path:
    """Return a Lance root, tolerating a single redundant same-name directory."""
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist after mount mapping: {path}")
    if (path / "_versions").is_dir():
        return path
    if path.is_dir():
        candidates = [
            child
            for child in path.iterdir()
            if child.is_dir() and (child / "_versions").is_dir()
        ]
        if len(candidates) == 1:
            print(f"[info] Lance root found one level below input: {candidates[0]}")
            return candidates[0]
    raise ValueError(
        f"Not a readable Lance root (missing _versions, and no unique nested Lance root): {path}"
    )


def shard_label(raw_input: str) -> str:
    if WINDOWS_DRIVE_PATH.match(raw_input):
        return Path(raw_input).name or raw_input
    parsed = urlparse(raw_input)
    name = PurePosixPath(parsed.path).name if parsed.scheme else Path(raw_input).name
    return name or raw_input


def inferred_dataset_label(raw_input: str) -> str:
    if WINDOWS_DRIVE_PATH.match(raw_input):
        parts = list(Path(raw_input).parts)
        parsed = None
    else:
        parsed = urlparse(raw_input)
        parts = list(PurePosixPath(parsed.path).parts) if parsed.scheme else list(Path(raw_input).parts)
    ignored = {"processed_parts", "parts", "dataset", "data"}
    for part in reversed(parts[:-1]):
        clean = str(part).strip("/\\")
        if clean and clean.lower() not in ignored and not clean.lower().endswith(".lance"):
            return clean
    return (parsed.netloc if parsed is not None else "") or "unknown_dataset"


def assign_dataset_overrides(inputs: Sequence[str], names: Sequence[str]) -> List[Optional[str]]:
    if not names:
        return [None] * len(inputs)
    if len(names) == 1:
        return [names[0]] * len(inputs)
    if len(names) == len(inputs):
        return list(names)
    raise ValueError(
        f"--dataset-name was supplied {len(names)} times for {len(inputs)} inputs; "
        "supply it once or once per input."
    )


@dataclass
class TailAccumulator:
    total_rows: int = 0
    tail_rows: int = 0
    no_eligible_rows: int = 0
    rows_with_excluded_constant_channels: int = 0
    eligible_channels: int = 0
    excluded_constant_channels: int = 0
    excluded_invalid_channels: int = 0
    tail_sample_counts: Counter = field(default_factory=Counter)

    def add(
        self,
        qualified_tail_samples: int,
        eligible_channels: int,
        constant_channels: int,
        invalid_channels: int,
    ) -> None:
        self.total_rows += 1
        self.eligible_channels += eligible_channels
        self.excluded_constant_channels += constant_channels
        self.excluded_invalid_channels += invalid_channels
        if eligible_channels == 0:
            self.no_eligible_rows += 1
        if constant_channels:
            self.rows_with_excluded_constant_channels += 1
        if qualified_tail_samples > 0:
            self.tail_rows += 1
            self.tail_sample_counts[int(qualified_tail_samples)] += 1

    def merge(self, other: "TailAccumulator") -> None:
        self.total_rows += other.total_rows
        self.tail_rows += other.tail_rows
        self.no_eligible_rows += other.no_eligible_rows
        self.rows_with_excluded_constant_channels += other.rows_with_excluded_constant_channels
        self.eligible_channels += other.eligible_channels
        self.excluded_constant_channels += other.excluded_constant_channels
        self.excluded_invalid_channels += other.excluded_invalid_channels
        self.tail_sample_counts.update(other.tail_sample_counts)

    def _nearest_rank(self, quantile: float) -> Optional[int]:
        count = sum(self.tail_sample_counts.values())
        if count == 0:
            return None
        target = max(1, math.ceil(quantile * count))
        running = 0
        for value, value_count in sorted(self.tail_sample_counts.items()):
            running += value_count
            if running >= target:
                return int(value)
        raise RuntimeError("quantile counter is inconsistent")

    def summary(
        self, sampling_rate: float, thresholds_seconds: Sequence[float]
    ) -> Dict[str, object]:
        def ratio(numerator: int) -> float:
            return numerator / self.total_rows if self.total_rows else 0.0

        out: Dict[str, object] = {
            "rows": self.total_rows,
            "rows_with_tail": self.tail_rows,
            "rows_with_tail_ratio": ratio(self.tail_rows),
            "rows_without_eligible_channels": self.no_eligible_rows,
            "rows_without_eligible_channels_ratio": ratio(self.no_eligible_rows),
            "rows_with_excluded_constant_channels": self.rows_with_excluded_constant_channels,
            "eligible_channels_total": self.eligible_channels,
            "excluded_constant_channels_total": self.excluded_constant_channels,
            "excluded_invalid_channels_total": self.excluded_invalid_channels,
        }
        for label, quantile in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
            samples = self._nearest_rank(quantile)
            out[f"tail_seconds_{label}"] = (
                samples / sampling_rate if samples is not None else None
            )
        maximum = max(self.tail_sample_counts, default=None)
        out["tail_seconds_max"] = maximum / sampling_rate if maximum is not None else None
        for threshold in thresholds_seconds:
            count = sum(
                value_count
                for samples, value_count in self.tail_sample_counts.items()
                if samples / sampling_rate > threshold
            )
            key = f"rows_tail_gt_{threshold:g}s"
            out[key] = count
            out[f"{key}_ratio"] = ratio(count)
        return out


def arrow_list_row_values(array: pa.Array, row_index: int) -> np.ndarray:
    """Extract one list row without constructing Python float objects."""
    scalar = array[row_index]
    if not scalar.is_valid:
        raise ValueError("signal column contains null")
    if isinstance(array, (pa.ListArray, pa.LargeListArray)):
        offsets = array.offsets.to_numpy(zero_copy_only=False)
        start, end = int(offsets[row_index]), int(offsets[row_index + 1])
        values = array.values.slice(start, end - start)
        if pa.types.is_floating(values.type) or pa.types.is_integer(values.type):
            return values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if isinstance(array, pa.FixedSizeListArray):
        size = array.type.list_size
        start = (array.offset + row_index) * size
        values = array.values.slice(start, size)
        if pa.types.is_floating(values.type) or pa.types.is_integer(values.type):
            return values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return np.asarray(scalar.as_py(), dtype=np.float32).reshape(-1)


def reconstruct_signal(
    flat_values: np.ndarray,
    n_channels: int,
    num_patches: int,
    patch_len: int,
    patch_stride_samples: Optional[int],
) -> np.ndarray:
    expected = n_channels * num_patches * patch_len
    if flat_values.size != expected:
        raise ValueError(
            f"token_inputs length {flat_values.size} != n_channels*num_patches*patch_len "
            f"({n_channels}*{num_patches}*{patch_len}={expected})"
        )
    patches = flat_values.reshape(n_channels, num_patches, patch_len)
    stride = patch_len if patch_stride_samples is None else patch_stride_samples
    if stride > patch_len:
        raise ValueError(
            f"patch stride {stride} exceeds patch_len {patch_len}; gaps cannot be reconstructed"
        )
    if stride == patch_len:
        return patches.reshape(n_channels, num_patches * patch_len)

    total_points = (num_patches - 1) * stride + patch_len
    signal = np.empty((n_channels, total_points), dtype=patches.dtype)
    for patch_index in range(num_patches):
        start = patch_index * stride
        signal[:, start : start + patch_len] = patches[:, patch_index, :]
    return signal


@dataclass(frozen=True)
class RowTail:
    shared_tail_samples: int
    eligible_channels: int
    constant_channels: int
    invalid_channels: int
    min_channel_tail_samples: int
    median_channel_tail_samples: float
    max_channel_tail_samples: int


def find_row_tail(
    signal: np.ndarray,
    valid_mask: Optional[np.ndarray],
    atol: float,
    rtol: float,
) -> RowTail:
    if signal.ndim != 2 or signal.shape[1] == 0:
        raise ValueError(f"Expected non-empty [channels, time] signal, got {signal.shape}")
    n_channels = signal.shape[0]
    if valid_mask is None:
        mask = np.ones(n_channels, dtype=bool)
    else:
        mask_values = np.asarray(valid_mask).reshape(-1)
        if mask_values.size != n_channels:
            raise ValueError(
                f"valid mask length {mask_values.size} != n_channels {n_channels}"
            )
        mask = np.isfinite(mask_values) & (mask_values > 0)

    tail_lengths: List[int] = []
    constant_channels = 0
    invalid_channels = int((~mask).sum())
    for channel_index in range(n_channels):
        if not mask[channel_index]:
            continue
        channel = signal[channel_index]
        final_value = channel[-1]
        if not np.isfinite(final_value):
            invalid_channels += 1
            continue
        equal_to_final = np.isclose(channel, final_value, atol=atol, rtol=rtol)
        if bool(equal_to_final.all()):
            constant_channels += 1
            continue
        last_different = int(np.flatnonzero(~equal_to_final)[-1])
        tail_lengths.append(channel.size - last_different - 1)

    if not tail_lengths:
        return RowTail(0, 0, constant_channels, invalid_channels, 0, 0.0, 0)
    lengths = np.asarray(tail_lengths, dtype=np.int64)
    shared = int(lengths.min())
    return RowTail(
        shared_tail_samples=shared,
        eligible_channels=len(tail_lengths),
        constant_channels=constant_channels,
        invalid_channels=invalid_channels,
        min_channel_tail_samples=shared,
        median_channel_tail_samples=float(np.median(lengths)),
        max_channel_tail_samples=int(lengths.max()),
    )


def scalar_column(batch: pa.RecordBatch, name: str, default: object) -> List[object]:
    if name not in batch.schema.names:
        return [default] * batch.num_rows
    return batch.column(batch.schema.get_field_index(name)).to_pylist()


def mask_for_row(batch: pa.RecordBatch, name: str, row_index: int) -> Optional[np.ndarray]:
    if name not in batch.schema.names:
        return None
    array = batch.column(batch.schema.get_field_index(name))
    if not array[row_index].is_valid:
        return None
    return arrow_list_row_values(array, row_index)


def normalize_text(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def requested_columns(schema: pa.Schema, args: argparse.Namespace) -> List[str]:
    required = [args.signal_column, "n_channels", "num_patches", "patch_len"]
    missing = [name for name in required if name not in schema.names]
    if missing:
        raise ValueError(f"Required columns missing from Lance schema: {missing}; schema={schema}")
    optional = [
        args.valid_mask_column,
        args.dataset_column,
        args.subject_column,
        args.sample_id_column,
    ]
    return required + [name for name in optional if name in schema.names and name not in required]


def iterate_limited_batches(
    dataset: object, columns: Sequence[str], batch_size: int, limit_rows: Optional[int]
) -> Iterator[pa.RecordBatch]:
    seen = 0
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
    for batch in scanner.to_batches():
        if limit_rows is None:
            yield batch
            continue
        remaining = limit_rows - seen
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        yield batch
        seen += batch.num_rows


def add_to_groups(
    groups: Mapping[str, object],
    dataset_name: str,
    shard_name: str,
    subject_name: str,
    qualified_samples: int,
    row_tail: RowTail,
) -> None:
    keys = (
        ("overall", ()),
        ("dataset", (dataset_name,)),
        ("shard", (dataset_name, shard_name)),
        ("subject", (dataset_name, subject_name)),
    )
    for group_type, key in keys:
        accumulator = groups[group_type][key]
        accumulator.add(
            qualified_samples,
            row_tail.eligible_channels,
            row_tail.constant_channels,
            row_tail.invalid_channels,
        )


def make_group_store() -> Dict[str, defaultdict]:
    return {
        "overall": defaultdict(TailAccumulator),
        "dataset": defaultdict(TailAccumulator),
        "shard": defaultdict(TailAccumulator),
        "subject": defaultdict(TailAccumulator),
    }


def flatten_summaries(
    groups: Mapping[str, Mapping[Tuple[str, ...], TailAccumulator]],
    sampling_rate: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    order = {"overall": 0, "dataset": 1, "shard": 2, "subject": 3}
    for group_type in ("overall", "dataset", "shard", "subject"):
        for key, accumulator in groups[group_type].items():
            dataset_name = key[0] if len(key) >= 1 else "<all>"
            group_name = key[-1] if key else "<all>"
            row = {
                "group_type": group_type,
                "dataset": dataset_name,
                "group_name": group_name,
            }
            row.update(accumulator.summary(sampling_rate, DEFAULT_THRESHOLDS_SECONDS))
            rows.append(row)
    rows.sort(key=lambda item: (order[str(item["group_type"])], str(item["dataset"]), str(item["group_name"])))
    return rows


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty summary")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def display_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_compact_summary(rows: Sequence[Mapping[str, object]]) -> None:
    print("\n=== Constant-tail summary ===")
    headers = ("type", "dataset", "group", "rows", "tail%", "p50s", "p90s", "p99s", "maxs", ">0.5s%", ">1s%", ">2s%")
    print("\t".join(headers))
    for row in rows:
        if row["group_type"] not in {"overall", "dataset", "shard"}:
            continue
        values = (
            row["group_type"],
            row["dataset"],
            row["group_name"],
            row["rows"],
            100.0 * float(row["rows_with_tail_ratio"]),
            row["tail_seconds_p50"],
            row["tail_seconds_p90"],
            row["tail_seconds_p99"],
            row["tail_seconds_max"],
            100.0 * float(row["rows_tail_gt_0.5s_ratio"]),
            100.0 * float(row["rows_tail_gt_1s_ratio"]),
            100.0 * float(row["rows_tail_gt_2s_ratio"]),
        )
        print("\t".join(display_value(value) for value in values))


def open_row_details(path: Optional[str]):
    if path is None:
        return None, None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("w", newline="", encoding="utf-8-sig")
    fields = [
        "input",
        "resolved_path",
        "dataset",
        "shard",
        "subject_id",
        "sample_id",
        "has_tail",
        "tail_samples",
        "tail_seconds",
        "eligible_channels",
        "excluded_constant_channels",
        "excluded_invalid_channels",
        "channel_tail_min_samples",
        "channel_tail_median_samples",
        "channel_tail_max_samples",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return handle, writer


def run(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    inputs = load_inputs(args.inputs, args.input_list)
    overrides = assign_dataset_overrides(inputs, args.dataset_name)
    min_tail_samples = max(1, math.ceil(args.min_tail_seconds * args.sampling_rate))
    groups = make_group_store()
    input_reports: List[Dict[str, object]] = []
    detail_handle, detail_writer = open_row_details(args.row_details)

    try:
        for input_index, raw_input in enumerate(inputs):
            resolved = map_input_path(raw_input, args.cos_mount_root, args.cos_include_bucket)
            shard = shard_label(raw_input)
            fallback_dataset = overrides[input_index] or inferred_dataset_label(raw_input)
            dataset = lance.dataset(str(resolved))
            columns = requested_columns(dataset.schema, args)
            available_rows = int(dataset.count_rows())
            expected_rows = min(available_rows, args.limit_rows) if args.limit_rows else available_rows
            print(
                f"\n[{input_index + 1}/{len(inputs)}] {raw_input}\n"
                f"  resolved={resolved}\n  rows={expected_rows}/{available_rows}; columns={columns}"
            )

            processed = 0
            signal_column_index = columns.index(args.signal_column)
            for batch in iterate_limited_batches(dataset, columns, args.batch_size, args.limit_rows):
                signals = batch.column(signal_column_index)
                n_channels_values = scalar_column(batch, "n_channels", None)
                num_patches_values = scalar_column(batch, "num_patches", None)
                patch_len_values = scalar_column(batch, "patch_len", None)
                dataset_values = scalar_column(batch, args.dataset_column, None)
                subject_values = scalar_column(batch, args.subject_column, None)
                sample_values = scalar_column(batch, args.sample_id_column, None)

                for row_index in range(batch.num_rows):
                    n_channels = int(n_channels_values[row_index])
                    num_patches = int(num_patches_values[row_index])
                    patch_len = int(patch_len_values[row_index])
                    flat = arrow_list_row_values(signals, row_index)
                    signal = reconstruct_signal(
                        flat,
                        n_channels,
                        num_patches,
                        patch_len,
                        args.patch_stride_samples,
                    )
                    valid_mask = mask_for_row(batch, args.valid_mask_column, row_index)
                    row_tail = find_row_tail(signal, valid_mask, args.atol, args.rtol)
                    qualified_samples = (
                        row_tail.shared_tail_samples
                        if row_tail.shared_tail_samples >= min_tail_samples
                        else 0
                    )

                    dataset_name = normalize_text(
                        overrides[input_index] if overrides[input_index] is not None else dataset_values[row_index],
                        fallback_dataset,
                    )
                    subject_name = normalize_text(subject_values[row_index], "<missing>")
                    add_to_groups(
                        groups,
                        dataset_name,
                        shard,
                        subject_name,
                        qualified_samples,
                        row_tail,
                    )
                    if detail_writer is not None:
                        detail_writer.writerow(
                            {
                                "input": raw_input,
                                "resolved_path": str(resolved),
                                "dataset": dataset_name,
                                "shard": shard,
                                "subject_id": subject_name,
                                "sample_id": normalize_text(sample_values[row_index], "<missing>"),
                                "has_tail": int(qualified_samples > 0),
                                "tail_samples": qualified_samples,
                                "tail_seconds": qualified_samples / args.sampling_rate,
                                "eligible_channels": row_tail.eligible_channels,
                                "excluded_constant_channels": row_tail.constant_channels,
                                "excluded_invalid_channels": row_tail.invalid_channels,
                                "channel_tail_min_samples": row_tail.min_channel_tail_samples,
                                "channel_tail_median_samples": row_tail.median_channel_tail_samples,
                                "channel_tail_max_samples": row_tail.max_channel_tail_samples,
                            }
                        )
                    processed += 1
                    if args.progress_every and processed % args.progress_every == 0:
                        print(f"  processed {processed}/{expected_rows}", flush=True)

            input_reports.append(
                {
                    "input": raw_input,
                    "resolved_path": str(resolved),
                    "shard": shard,
                    "available_rows": available_rows,
                    "processed_rows": processed,
                }
            )
    finally:
        if detail_handle is not None:
            detail_handle.close()

    summary_rows = flatten_summaries(groups, args.sampling_rate)
    report = {
        "definition": (
            "For each valid channel, scan backward for values equal to its final value. "
            "Exclude whole-recording-constant channels. A row tail is the minimum suffix "
            "length across all remaining channels, and is counted only when it reaches "
            "min_tail_seconds. Ratios use all processed rows as denominator; tail quantiles "
            "use only rows that meet the minimum. Overall results are accumulated row by row "
            "and are therefore weighted by actual row count."
        ),
        "config": {
            "sampling_rate": args.sampling_rate,
            "min_tail_seconds": args.min_tail_seconds,
            "min_tail_samples": min_tail_samples,
            "atol": args.atol,
            "rtol": args.rtol,
            "patch_stride_samples": args.patch_stride_samples,
            "thresholds_seconds_strictly_greater_than": list(DEFAULT_THRESHOLDS_SECONDS),
            "limit_rows_per_input": args.limit_rows,
        },
        "inputs": input_reports,
        "summaries": summary_rows,
    }
    return summary_rows, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        summary_rows, report = run(args)
        prefix = Path(args.output_prefix)
        summary_path = Path(f"{prefix}.summary.csv")
        report_path = Path(f"{prefix}.report.json")
        write_summary_csv(summary_path, summary_rows)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print_compact_summary(summary_rows)
        print(f"\nWrote summary: {summary_path.resolve()}")
        print(f"Wrote report:  {report_path.resolve()}")
        if args.row_details:
            print(f"Wrote rows:    {Path(args.row_details).resolve()}")
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
