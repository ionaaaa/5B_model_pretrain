#!/usr/bin/env python3
"""Read-only physical/semantic schema probe for mounted canonical EEG Lance data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence

import lance
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runtime_adaptation.inria import (  # noqa: E402
    INRIA_CHANNELS,
    normalize_integer_scalar,
)
from runtime_adaptation.canonical import parse_canonical_schema_metadata  # noqa: E402


DEFAULT_DATASETS = ("inria", "egobrain", "shin2017a")
SHAPE_FIELDS = ("eeg_shape", "data_shape", "shape")
CHANNEL_COUNT_FIELDS = ("eeg_n_channels", "n_channels", "channel_count")
CHANNEL_INDEX_FIELDS = ("eeg_channel_indices", "channel_indices")
CHANNEL_NAME_FIELDS = (
    "eeg_channel_names",
    "channel_names",
    "ch_names",
    "channels",
)
VALID_SAMPLE_FIELDS = ("valid_samples", "valid_time_samples", "valid_length")
SFREQ_FIELDS = (
    "sfreq",
    "eeg_sfreq",
    "sampling_rate",
    "sample_rate",
    "fs",
)
IDENTITY_FIELDS = ("dataset_id", "dataset_key", "subject", "source_index", "sample_id")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read mounted canonical Lance datasets without writing data and report "
            "physical schema plus sampled semantic fingerprints."
        )
    )
    parser.add_argument(
        "--cos-mount-root",
        help="Mounted COS root containing datasets/<key>/releases/<release>/data.lance",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset keys resolved below --cos-mount-root",
    )
    parser.add_argument("--release", default="pretrain_V1")
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Explicit mounted path; overrides the generated path for KEY",
    )
    parser.add_argument("--head-rows", type=int, default=4)
    parser.add_argument(
        "--max-scan-rows",
        type=int,
        default=100000,
        help="Maximum rows for distributions; 0 scans the full dataset",
    )
    parser.add_argument("--scan-batch-size", type=int, default=8192)
    return parser.parse_args()


def _decode_metadata(metadata) -> Dict[str, str]:
    if not metadata:
        return {}
    decoded = {}
    for key, value in metadata.items():
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="replace")
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        decoded[str(key)] = str(value)
    return decoded


def _first_existing(names: Sequence[str], available: Iterable[str]) -> Optional[str]:
    available = set(available)
    return next((name for name in names if name in available), None)


def _decode_sequence(value: Any):
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return None
        else:
            return None
    if isinstance(value, dict):
        for key in ("channel_names", "channels", "values", "indices"):
            if key in value:
                return _decode_sequence(value[key])
        return None
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _value_summary(value: Any) -> str:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None:
        return "null"
    if isinstance(value, (bytes, bytearray)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, str):
        return repr(value if len(value) <= 120 else value[:117] + "...")
    sequence = _decode_sequence(value)
    if sequence is not None:
        preview = sequence[:8]
        return f"{type(value).__name__}(len={len(sequence)}, head={preview!r})"
    return repr(value)


def _safe_dataset_version(dataset):
    value = getattr(dataset, "version", None)
    try:
        return value() if callable(value) else value
    except Exception as exc:  # read-only diagnostic, preserve the failure text
        return f"UNAVAILABLE ({type(exc).__name__}: {exc})"


def _scan_semantics(dataset, max_scan_rows: int, batch_size: int):
    names = dataset.schema.names
    selected = {
        "shape": _first_existing(SHAPE_FIELDS, names),
        "channel_count": _first_existing(CHANNEL_COUNT_FIELDS, names),
        "channel_indices": _first_existing(CHANNEL_INDEX_FIELDS, names),
        "channel_names": _first_existing(CHANNEL_NAME_FIELDS, names),
        "valid_samples": _first_existing(VALID_SAMPLE_FIELDS, names),
        "sfreq": _first_existing(SFREQ_FIELDS, names),
    }
    columns = list(dict.fromkeys(field for field in selected.values() if field))
    if not columns:
        return {
            "selected_fields": selected,
            "scanned_rows": 0,
            "shape_counts": [],
            "channel_count_counts": [],
            "valid_samples_counts": [],
            "valid_samples_raw_types": [],
            "valid_samples_unparseable": 0,
            "sfreq_counts": [],
            "observed_channel_indices": [],
            "observed_channel_names": [],
        }

    shape_counts = Counter()
    channel_count_counts = Counter()
    valid_counts = Counter()
    valid_raw_types = Counter()
    valid_unparseable = 0
    sfreq_counts = Counter()
    channel_indices = set()
    channel_names = set()
    scanned = 0
    limit = None if max_scan_rows == 0 else max_scan_rows

    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        take = batch.num_rows if limit is None else min(batch.num_rows, limit - scanned)
        if take <= 0:
            break
        rows = batch.slice(0, take).to_pylist()
        for row in rows:
            if selected["shape"]:
                value = _decode_sequence(row.get(selected["shape"]))
                if value is not None:
                    shape_counts[tuple(value)] += 1
            if selected["channel_count"]:
                value = normalize_integer_scalar(row.get(selected["channel_count"]))
                if value is not None:
                    channel_count_counts[value] += 1
            if selected["valid_samples"]:
                raw = row.get(selected["valid_samples"])
                valid_raw_types[type(raw).__name__] += 1
                value = normalize_integer_scalar(raw)
                if value is None:
                    valid_unparseable += 1
                else:
                    valid_counts[value] += 1
            if selected["sfreq"]:
                raw = row.get(selected["sfreq"])
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    value = None
                if value is not None and np.isfinite(value):
                    sfreq_counts[value] += 1
            if selected["channel_indices"]:
                value = _decode_sequence(row.get(selected["channel_indices"]))
                if value is not None:
                    for item in value:
                        integer = normalize_integer_scalar(item)
                        if integer is not None:
                            channel_indices.add(integer)
            if selected["channel_names"]:
                value = _decode_sequence(row.get(selected["channel_names"]))
                if value is not None:
                    channel_names.update(str(item) for item in value)
        scanned += take
        if limit is not None and scanned >= limit:
            break

    return {
        "selected_fields": selected,
        "scanned_rows": scanned,
        "shape_counts": shape_counts.most_common(20),
        "channel_count_counts": channel_count_counts.most_common(20),
        "valid_samples_counts": valid_counts.most_common(30),
        "valid_samples_raw_types": valid_raw_types.most_common(10),
        "valid_samples_unparseable": valid_unparseable,
        "sfreq_counts": sfreq_counts.most_common(20),
        "observed_channel_indices": sorted(channel_indices),
        "observed_channel_names": sorted(channel_names),
    }


def probe_dataset(key: str, path: str, head_rows: int, max_scan_rows: int, batch_size: int):
    dataset = lance.dataset(path)
    schema = dataset.schema
    row_count = dataset.count_rows()
    try:
        fragment_count = len(list(dataset.get_fragments()))
    except Exception as exc:
        fragment_count = f"UNAVAILABLE ({type(exc).__name__}: {exc})"

    fields = [
        {
            "name": field.name,
            "physical_type": str(field.type),
            "nullable": field.nullable,
            "metadata": _decode_metadata(field.metadata),
        }
        for field in schema
    ]
    fingerprint = tuple(
        (field["name"], field["physical_type"], field["nullable"])
        for field in fields
    )
    semantics = _scan_semantics(dataset, max_scan_rows, batch_size)

    head_count = min(head_rows, row_count)
    head_columns = [
        name for name in (
            "eeg", *SHAPE_FIELDS, *CHANNEL_COUNT_FIELDS, *CHANNEL_INDEX_FIELDS,
            *CHANNEL_NAME_FIELDS, *VALID_SAMPLE_FIELDS, *SFREQ_FIELDS, *IDENTITY_FIELDS,
        )
        if name in schema.names
    ]
    head = []
    if head_count and head_columns:
        rows = dataset.take(list(range(head_count)), columns=head_columns).to_pylist()
        for row_index, row in enumerate(rows):
            head.append(
                {
                    "row": row_index,
                    "fields": {name: _value_summary(row.get(name)) for name in head_columns},
                }
            )

    direct_sfreq = semantics["sfreq_counts"]
    direct_catalog = semantics["observed_channel_names"]
    metadata_sfreq = None
    metadata_catalogs = []
    try:
        canonical_metadata = parse_canonical_schema_metadata(schema)
        metadata_sfreq = float(canonical_metadata["recipe"]["sampling_rates"]["eeg"])
        metadata_catalogs = [
            {
                "channel_map_id": str(entry["channel_map_id"]),
                "channel_names": list(entry["channel_names"]),
            }
            for entry in canonical_metadata["channel_metadata_catalog"]["eeg"]
        ]
    except (KeyError, TypeError, ValueError):
        pass
    registry_sfreq_hint = 100.0 if key == "inria" else None
    registry_catalog_hint = list(INRIA_CHANNELS) if key == "inria" else None
    return {
        "key": key,
        "path": path,
        "row_count": row_count,
        "dataset_version": _safe_dataset_version(dataset),
        "fragment_count": fragment_count,
        "schema_metadata": _decode_metadata(schema.metadata),
        "fields": fields,
        "fingerprint": fingerprint,
        "head": head,
        "semantics": semantics,
        "sampling_rate_evidence": {
            "physical_field_values": direct_sfreq,
            "schema_metadata_value": metadata_sfreq,
            "registry_hint": registry_sfreq_hint,
            "established_from_lance": bool(direct_sfreq) or metadata_sfreq is not None,
        },
        "channel_catalog_evidence": {
            "physical_channel_names": direct_catalog,
            "schema_metadata_catalogs": metadata_catalogs,
            "observed_channel_indices": semantics["observed_channel_indices"],
            "registry_hint": registry_catalog_hint,
            "established_from_lance": bool(direct_catalog) or bool(metadata_catalogs),
        },
    }


def _print_report(report):
    print("\n" + "=" * 88)
    print(f"DATASET: {report['key']}")
    print(f"path: {report['path']}")
    print(
        f"row_count={report['row_count']} version={report['dataset_version']} "
        f"fragments={report['fragment_count']}"
    )
    print("schema_metadata:", report["schema_metadata"] or "{}")
    print("physical_schema:")
    for field in report["fields"]:
        print(
            f"  - {field['name']}: {field['physical_type']} "
            f"nullable={field['nullable']} metadata={field['metadata'] or '{}'}"
        )
    print("head_rows:")
    for row in report["head"]:
        print(f"  row {row['row']}:")
        for name, summary in row["fields"].items():
            print(f"    {name}: {summary}")
    semantics = report["semantics"]
    print("semantic_scan:")
    print(f"  scanned_rows: {semantics['scanned_rows']}")
    print(f"  selected_fields: {semantics['selected_fields']}")
    print(f"  eeg_shape distribution: {semantics['shape_counts']}")
    print(f"  channel_count distribution: {semantics['channel_count_counts']}")
    print(f"  valid_samples distribution: {semantics['valid_samples_counts']}")
    print(f"  valid_samples raw types: {semantics['valid_samples_raw_types']}")
    print(f"  valid_samples unparseable: {semantics['valid_samples_unparseable']}")
    print("sampling_rate_evidence:", report["sampling_rate_evidence"])
    print("channel_catalog_evidence:", report["channel_catalog_evidence"])


def _print_comparison(reports):
    print("\n" + "=" * 88)
    print("CROSS-DATASET PHYSICAL SCHEMA COMPARISON")
    fingerprints = {report["key"]: report["fingerprint"] for report in reports}
    first = reports[0]["fingerprint"] if reports else ()
    print("all_exact_physical_schemas_equal:", all(r["fingerprint"] == first for r in reports))
    all_names = {report["key"]: {field[0] for field in report["fingerprint"]} for report in reports}
    common = set.intersection(*all_names.values()) if all_names else set()
    print("common_fields:", sorted(common))
    for key, names in all_names.items():
        print(f"fields_only_or_not_common[{key}]:", sorted(names - common))
    for left_index, left in enumerate(reports):
        for right in reports[left_index + 1:]:
            print(
                f"exact_schema_equal[{left['key']} vs {right['key']}]:",
                fingerprints[left["key"]] == fingerprints[right["key"]],
            )


def _resolve_paths(args):
    explicit = {}
    for item in args.dataset_path:
        if "=" not in item:
            raise ValueError(f"--dataset-path must be KEY=PATH, got {item!r}")
        key, path = item.split("=", 1)
        explicit[key.strip()] = path.strip()
    if not args.cos_mount_root and not explicit:
        raise ValueError("Provide --cos-mount-root or at least one --dataset-path KEY=PATH")

    resolved = {}
    for key in args.datasets:
        if key in explicit:
            resolved[key] = explicit[key]
        elif args.cos_mount_root:
            resolved[key] = os.path.join(
                args.cos_mount_root,
                "datasets",
                key,
                "releases",
                args.release,
                "data.lance",
            )
    for key, path in explicit.items():
        resolved.setdefault(key, path)
    return resolved


def main():
    args = parse_args()
    if args.head_rows < 0 or args.max_scan_rows < 0 or args.scan_batch_size <= 0:
        raise ValueError("invalid probe row/batch settings")
    paths = _resolve_paths(args)
    reports = []
    failures = []
    for key, path in paths.items():
        if not os.path.exists(path):
            failures.append(f"{key}: mounted path does not exist: {path}")
            continue
        try:
            report = probe_dataset(
                key,
                path,
                head_rows=args.head_rows,
                max_scan_rows=args.max_scan_rows,
                batch_size=args.scan_batch_size,
            )
        except Exception as exc:
            failures.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        reports.append(report)
        _print_report(report)
    if reports:
        _print_comparison(reports)
    print("\nREAD_ONLY: no Lance dataset was written or modified.")
    if failures:
        raise RuntimeError("Schema probe failures: " + " | ".join(failures))


if __name__ == "__main__":
    main()
