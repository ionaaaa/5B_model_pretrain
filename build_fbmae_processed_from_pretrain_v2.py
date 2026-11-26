"""Convert canonical pretrain_V2 Lance windows to the legacy FBMAE row schema.

The converter is intentionally local-filesystem only.  On Tencent Cloud the COS
bucket is expected to be mounted (for example below /opt/ml/omni-eeg-processed).
It never opens COS through S3 and never mutates the input datasets.

Signal order for every row is fixed and audited:

    decode only the real ``valid_samples``
    -> map source channels to the legacy A1/A2 64-channel vocabulary
    -> anti-aliased resample to 100 Hz
    -> z-score each present channel over real samples only
    -> right-pad the z-scored waveform to 1000 samples with exact zeros
    -> compute FBMAE time-frequency targets from that unmasked waveform
    -> channel-major patch/token flattening

The output Arrow schema deliberately matches the historical
``processed_parts/part_*.lance`` physical schema: variable list<double> for
floating arrays, variable list<int64> for indices, and no ``label`` column.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import lance
import numpy as np
import pyarrow as pa
from scipy.signal import resample_poly, spectrogram


SOURCE_SFREQ = 200.0
TARGET_SFREQ = 100.0
WINDOW_SECONDS = 10.0
TOTAL_POINTS = 1000
PATCH_LEN = 100
NUM_TIME_PATCHES = 10
N_CHANNELS = 64
NUM_TOKENS = N_CHANNELS * NUM_TIME_PATCHES
N_BANDS = 5
FRAMES_PER_PATCH = 5
EPS = 1e-6

BAND_DEFS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma_low", 30.0, 45.0),
)

# This is the vocabulary used by the exact generator of the reference part.
# Do not replace A1/A2 with F9/F10 without creating a new data contract.
STANDARD_64_CHANNELS: Tuple[str, ...] = (
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz", "A1", "A2",
)
assert len(STANDARD_64_CHANNELS) == N_CHANNELS

ALIASES = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


# pa.list_(...) without list_size is intentional: it reproduces the physical
# schema of the reference part rather than the newer fixed-size-list writer.
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("num_tokens", pa.int64()),
        pa.field("patch_len", pa.int64()),
        pa.field("n_bands", pa.int64()),
        pa.field("frames_per_patch", pa.int64()),
        pa.field("n_channels", pa.int64()),
        pa.field("num_patches", pa.int64()),
        pa.field("token_inputs", pa.list_(pa.float64())),
        pa.field("targets", pa.list_(pa.float64())),
        pa.field("token_channel_indices", pa.list_(pa.int64())),
        pa.field("token_time_indices", pa.list_(pa.int64())),
        pa.field("token_valid_mask", pa.list_(pa.float64())),
        pa.field("channel_valid_mask", pa.list_(pa.float64())),
        pa.field("sample_id", pa.int64()),
        pa.field("global_idx", pa.int64()),
        pa.field("subject_id", pa.string()),
        pa.field("edf_relpath", pa.string()),
        pa.field("segment_start_sec", pa.float64()),
        pa.field("split", pa.string()),
        pa.field("shard_name", pa.string()),
        pa.field("dataset_key", pa.string()),
        pa.field("valid_channel_count", pa.int64()),
    ]
)


@dataclass(frozen=True)
class InputSpec:
    dataset_key: str
    lance_path: str
    catalog_path: Optional[str] = None
    channel_map_path: Optional[str] = None


@dataclass
class ChannelPreflight:
    rows_scanned: int = 0
    catalog_sources: Counter = field(default_factory=Counter)
    catalog_signatures: set[Tuple[str, ...]] = field(default_factory=set)
    source_to_standard: Dict[str, str] = field(default_factory=dict)
    unmapped_channels: Counter = field(default_factory=Counter)
    duplicate_destinations: Counter = field(default_factory=Counter)
    valid_channel_counts: Counter = field(default_factory=Counter)
    zero_mapped_rows: int = 0

    @property
    def eligible(self) -> bool:
        return (
            not self.duplicate_destinations
            and self.zero_mapped_rows == 0
        )


@dataclass
class ConversionStats:
    rows: int = 0
    tail_rows: int = 0
    input_valid_samples: Counter = field(default_factory=Counter)
    output_valid_samples: Counter = field(default_factory=Counter)
    valid_channel_counts: Counter = field(default_factory=Counter)
    dropped_channels: Counter = field(default_factory=Counter)
    duplicate_destinations: Counter = field(default_factory=Counter)
    processing_seconds: float = 0.0


def _decode_metadata(schema: pa.Schema) -> Dict[str, Any]:
    decoded: Dict[str, Any] = {}
    for raw_key, raw_value in (schema.metadata or {}).items():
        key = raw_key.decode("utf-8", errors="replace") if isinstance(raw_key, bytes) else str(raw_key)
        value = raw_value.decode("utf-8", errors="replace") if isinstance(raw_value, bytes) else raw_value
        try:
            decoded[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            decoded[key] = value
    return decoded


def _find_channel_names(value: Any) -> Optional[List[str]]:
    """Find a single ordered channel catalog in decoded schema metadata."""
    if isinstance(value, Mapping):
        for key in ("channel_names", "channel_catalog", "channels", "ch_names"):
            candidate = value.get(key)
            if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
                return [str(x) for x in candidate]
        for child in value.values():
            result = _find_channel_names(child)
            if result is not None:
                return result
    return None


class CatalogResolver:
    """Resolve the ordered source-channel list without guessing."""

    def __init__(self, schema: pa.Schema, catalog_path: Optional[str]) -> None:
        self.schema_catalog = _find_channel_names(_decode_metadata(schema))
        self.default_catalog: Optional[List[str]] = None
        self.by_recording: Dict[str, List[str]] = {}
        if catalog_path:
            with open(catalog_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._load(payload)

    def _load(self, payload: Any) -> None:
        if isinstance(payload, list) and all(isinstance(x, str) for x in payload):
            self.default_catalog = [str(x) for x in payload]
            return
        if not isinstance(payload, Mapping):
            raise ValueError("channel catalog JSON must be a list or object")

        direct = _find_channel_names(payload)
        if direct is not None and "recordings" not in payload:
            self.default_catalog = direct

        recordings = payload.get("recordings")
        if isinstance(recordings, list):
            for item in recordings:
                if not isinstance(item, Mapping):
                    continue
                key = item.get("recording_id") or item.get("path")
                names = item.get("channel_names")
                if key and isinstance(names, list) and all(isinstance(x, str) for x in names):
                    self.by_recording[str(key)] = [str(x) for x in names]

        # Also accept {"recording_id": ["Fp1", ...], ...}.
        for key, value in payload.items():
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                if key not in {"channel_names", "channel_catalog", "channels", "ch_names"}:
                    self.by_recording[str(key)] = [str(x) for x in value]

    def resolve(self, row: Mapping[str, Any]) -> Tuple[List[str], str]:
        row_names = row.get("channel_names")
        if isinstance(row_names, list) and all(isinstance(x, str) for x in row_names):
            return [str(x) for x in row_names], "row.channel_names"

        recording_id = str(row.get("recording_id", ""))
        if recording_id in self.by_recording:
            return self.by_recording[recording_id], "catalog.recording_id"
        if self.schema_catalog is not None:
            return list(self.schema_catalog), "schema.metadata"
        if self.default_catalog is not None:
            return list(self.default_catalog), "catalog.default"
        raise ValueError(
            f"No ordered channel catalog for recording {recording_id!r}. "
            "Supply --catalog DATASET_KEY=/path/to/preflight-or-catalog.json."
        )


def load_channel_override(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "source_to_standard" in payload:
        payload = payload["source_to_standard"]
    if not isinstance(payload, Mapping):
        raise ValueError("channel map JSON must be an object")
    result: Dict[str, str] = {}
    for source, destination in payload.items():
        if destination is None or str(destination).strip() == "":
            continue
        result[str(source).strip().casefold()] = str(destination).strip()
    return result


def canonical_channel_name(name: str, overrides: Mapping[str, str]) -> Optional[str]:
    raw = str(name).strip()
    override = overrides.get(raw.casefold())
    if override is not None:
        raw = override
    if raw.upper().startswith("EEG "):
        raw = raw[4:].strip()
    upper = raw.upper()
    for suffix in ("-REF", "-LE", "-A1", "-A2", "-M1", "-M2"):
        if upper.endswith(suffix):
            raw = raw[: -len(suffix)].strip()
            upper = raw.upper()
            break
    raw = ALIASES.get(upper, raw)
    lookup = {channel.casefold(): channel for channel in STANDARD_64_CHANNELS}
    return lookup.get(raw.casefold())


def align_to_standard_channels(
    signal: np.ndarray,
    source_names: Sequence[str],
    source_mask: np.ndarray,
    overrides: Mapping[str, str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if signal.ndim != 2:
        raise ValueError(f"EEG must be [C,T], got {signal.shape}")
    if len(source_names) != signal.shape[0] or source_mask.shape != (signal.shape[0],):
        raise ValueError("channel catalog/mask length does not match EEG channel axis")

    destination_lookup = {name: idx for idx, name in enumerate(STANDARD_64_CHANNELS)}
    sums = np.zeros((N_CHANNELS, signal.shape[1]), dtype=np.float32)
    counts = np.zeros(N_CHANNELS, dtype=np.int64)
    dropped: List[str] = []
    duplicates: List[str] = []
    for source_idx, source_name in enumerate(source_names):
        if not bool(source_mask[source_idx]):
            continue
        destination = canonical_channel_name(source_name, overrides)
        if destination is None:
            dropped.append(str(source_name))
            continue
        destination_idx = destination_lookup[destination]
        sums[destination_idx] += signal[source_idx]
        counts[destination_idx] += 1
        if counts[destination_idx] == 2:
            duplicates.append(destination)

    valid = counts > 0
    aligned = np.zeros_like(sums)
    aligned[valid] = sums[valid] / counts[valid, None]
    return aligned, valid.astype(np.float32), dropped, duplicates


def resample_valid_signal(signal: np.ndarray, source_sfreq: float) -> Tuple[np.ndarray, int]:
    if source_sfreq <= 0:
        raise ValueError(f"invalid sample rate {source_sfreq}")
    expected = int(round(signal.shape[1] * TARGET_SFREQ / source_sfreq))
    source_i = int(round(source_sfreq))
    target_i = int(round(TARGET_SFREQ))
    if not math.isclose(source_sfreq, source_i, abs_tol=1e-6):
        raise ValueError(f"non-integer source sample rate is not supported: {source_sfreq}")
    gcd = math.gcd(source_i, target_i)
    output = resample_poly(signal, up=target_i // gcd, down=source_i // gcd, axis=1).astype(np.float32)
    # scipy returns ceil(N*up/down); use duration-derived round for the row contract.
    if output.shape[1] > expected:
        output = output[:, :expected]
    elif output.shape[1] < expected:
        output = np.pad(output, ((0, 0), (0, expected - output.shape[1])))
    return output.astype(np.float32), expected


def zscore_valid_then_pad(
    signal: np.ndarray,
    channel_valid_mask: np.ndarray,
    valid_samples: int,
    eps: float = 1e-8,
) -> np.ndarray:
    if valid_samples <= 0 or valid_samples > TOTAL_POINTS:
        raise ValueError(f"valid samples at 100 Hz must be in [1,{TOTAL_POINTS}], got {valid_samples}")
    if signal.shape != (N_CHANNELS, valid_samples):
        raise ValueError(f"unexpected valid signal shape {signal.shape}")
    output = np.zeros((N_CHANNELS, TOTAL_POINTS), dtype=np.float32)
    valid_channels = channel_valid_mask.astype(bool)
    if valid_channels.any():
        real = signal[valid_channels, :valid_samples]
        mean = real.mean(axis=1, keepdims=True)
        std = real.std(axis=1, keepdims=True)
        output[valid_channels, :valid_samples] = (real - mean) / (std + eps)
    # Missing channels and temporal padding remain exact zero.
    return output


def compute_fbmae_targets(waveform: np.ndarray, channel_valid_mask: np.ndarray) -> np.ndarray:
    """Reproduce the reference Lance target algorithm, returning [64,10,5,5]."""
    if waveform.shape != (N_CHANNELS, TOTAL_POINTS):
        raise ValueError(f"waveform must be [64,1000], got {waveform.shape}")
    frequencies, stft_times, psd = spectrogram(
        waveform,
        fs=TARGET_SFREQ,
        window="hann",
        nperseg=100,
        noverlap=90,
        nfft=128,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="psd",
        axis=1,
    )
    bands: List[np.ndarray] = []
    for _, low, high in BAND_DEFS:
        selected = (frequencies >= low) & (frequencies < high)
        band = psd[:, selected, :].mean(axis=1)
        bands.append(np.log(band + EPS).astype(np.float32))
    band_power = np.stack(bands, axis=1).astype(np.float32)  # [C,F,S]

    sample_times = np.arange(TOTAL_POINTS, dtype=np.float32) / TARGET_SFREQ
    trajectory = np.zeros((N_CHANNELS, N_BANDS, TOTAL_POINTS), dtype=np.float32)
    for channel in range(N_CHANNELS):
        for band in range(N_BANDS):
            trajectory[channel, band] = np.interp(
                sample_times,
                stft_times,
                band_power[channel, band],
                left=band_power[channel, band, 0],
                right=band_power[channel, band, -1],
            )

    normalized = np.zeros_like(trajectory)
    valid = channel_valid_mask.astype(bool)
    if valid.any():
        real = trajectory[valid]
        normalized[valid] = (
            (real - real.mean(axis=2, keepdims=True))
            / (real.std(axis=2, keepdims=True) + 1e-8)
        )

    frame_offsets = np.linspace(0, PATCH_LEN - 1, FRAMES_PER_PATCH).astype(np.int64)
    targets = np.zeros(
        (N_CHANNELS, NUM_TIME_PATCHES, N_BANDS, FRAMES_PER_PATCH),
        dtype=np.float32,
    )
    for time_index in range(NUM_TIME_PATCHES):
        targets[:, time_index] = normalized[:, :, time_index * PATCH_LEN + frame_offsets]
    return targets


def _subject_from_recording_id(recording_id: str) -> str:
    match = re.search(r"(?:^|/)sub-([^/]+)", recording_id, flags=re.IGNORECASE)
    return match.group(1) if match else recording_id


def convert_row(
    row: Mapping[str, Any],
    dataset_key: str,
    global_idx: int,
    resolver: CatalogResolver,
    overrides: Mapping[str, str],
    stats: ConversionStats,
    allow_unmapped_channels: bool = True,
) -> Dict[str, Any]:
    started = time.perf_counter()
    raw_eeg = row.get("eeg")
    if raw_eeg is None:
        raise ValueError("row is missing eeg")
    eeg = np.asarray(raw_eeg, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be rectangular [C,T], got {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError("eeg contains NaN or Inf")

    source_sfreq = float(row.get("sample_rate_hz", SOURCE_SFREQ))
    if not math.isclose(source_sfreq, SOURCE_SFREQ, abs_tol=1e-6):
        raise ValueError(f"expected canonical 200 Hz input, got {source_sfreq}")
    valid_samples = int(row.get("valid_samples", eeg.shape[1]))
    max_source_samples = int(round(WINDOW_SECONDS * source_sfreq))
    if valid_samples <= 0 or valid_samples > max_source_samples or valid_samples > eeg.shape[1]:
        raise ValueError(
            f"invalid valid_samples={valid_samples}, eeg T={eeg.shape[1]}, max={max_source_samples}"
        )
    eeg = eeg[:, :valid_samples]

    channel_count = int(row.get("channel_count", eeg.shape[0]))
    if channel_count != eeg.shape[0]:
        raise ValueError(f"channel_count={channel_count} but eeg has {eeg.shape[0]} channels")
    source_mask = np.asarray(row.get("channel_mask", [True] * channel_count), dtype=bool)
    source_names, _ = resolver.resolve(row)
    aligned, channel_valid_mask, dropped, duplicates = align_to_standard_channels(
        eeg, source_names, source_mask, overrides
    )
    if duplicates or (dropped and not allow_unmapped_channels):
        raise ValueError(
            "channel mapping policy failed: "
            f"unmapped={sorted(set(dropped))}, "
            f"duplicate_destinations={sorted(set(duplicates))}"
        )
    if channel_valid_mask.sum() == 0:
        raise ValueError(
            "zero source channels map to the standard 64-channel vocabulary; "
            "an authoritative --channel-map is required"
        )

    resampled, valid_at_100hz = resample_valid_signal(aligned, source_sfreq)
    waveform = zscore_valid_then_pad(resampled, channel_valid_mask, valid_at_100hz)
    targets = compute_fbmae_targets(waveform, channel_valid_mask)

    patches = waveform.reshape(N_CHANNELS, NUM_TIME_PATCHES, PATCH_LEN)
    token_inputs = patches.reshape(NUM_TOKENS, PATCH_LEN)
    token_targets = targets.reshape(NUM_TOKENS, N_BANDS, FRAMES_PER_PATCH)
    token_channel_indices = np.repeat(np.arange(N_CHANNELS), NUM_TIME_PATCHES).astype(np.int64)
    token_time_indices = np.tile(np.arange(NUM_TIME_PATCHES), N_CHANNELS).astype(np.int64)
    token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)

    # Hard invariants used both as corruption checks and token-order checks.
    assert token_inputs.shape == (640, 100)
    assert token_targets.shape == (640, 5, 5)
    assert np.array_equal(token_channel_indices[:20], np.r_[np.zeros(10), np.ones(10)])
    assert np.array_equal(token_time_indices[:20], np.r_[np.arange(10), np.arange(10)])
    assert np.isfinite(token_inputs).all() and np.isfinite(token_targets).all()
    assert np.all(token_inputs[channel_valid_mask[token_channel_indices] == 0] == 0)
    if valid_at_100hz < TOTAL_POINTS:
        padded = waveform[channel_valid_mask.astype(bool), valid_at_100hz:]
        assert np.all(padded == 0), "temporal padding must remain exact zero"

    recording_id = str(row.get("recording_id", ""))
    stats.rows += 1
    stats.tail_rows += int(valid_at_100hz < TOTAL_POINTS)
    stats.input_valid_samples[valid_samples] += 1
    stats.output_valid_samples[valid_at_100hz] += 1
    stats.valid_channel_counts[int(channel_valid_mask.sum())] += 1
    stats.dropped_channels.update(dropped)
    stats.duplicate_destinations.update(duplicates)
    stats.processing_seconds += time.perf_counter() - started

    # Cast arrays to Python lists under OUTPUT_SCHEMA.  Arrow then writes the
    # same list<double>/list<int64> physical representation as the reference.
    return {
        "num_tokens": NUM_TOKENS,
        "patch_len": PATCH_LEN,
        "n_bands": N_BANDS,
        "frames_per_patch": FRAMES_PER_PATCH,
        "n_channels": N_CHANNELS,
        "num_patches": NUM_TIME_PATCHES,
        "token_inputs": token_inputs.astype(np.float64).reshape(-1).tolist(),
        "targets": token_targets.astype(np.float64).reshape(-1).tolist(),
        "token_channel_indices": token_channel_indices.tolist(),
        "token_time_indices": token_time_indices.tolist(),
        "token_valid_mask": token_valid_mask.astype(np.float64).tolist(),
        "channel_valid_mask": channel_valid_mask.astype(np.float64).tolist(),
        "sample_id": int(global_idx),
        "global_idx": int(global_idx),
        "subject_id": _subject_from_recording_id(recording_id),
        "edf_relpath": recording_id,
        "segment_start_sec": float(row.get("start_seconds", 0.0)),
        "split": "train",
        "shard_name": f"{dataset_key}:pretrain_V2",
        "dataset_key": dataset_key,
        "valid_channel_count": int(channel_valid_mask.sum()),
    }


class PartWriter:
    def __init__(
        self,
        output_dir: str,
        total_rows: int,
        rows_per_part: int,
        overwrite: bool,
        dry_run: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.total_rows = total_rows
        self.rows_per_part = rows_per_part
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.started_parts: set[str] = set()
        self.part_counts: Counter = Counter()
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, total_rows, rows_per_part):
                path = self._path(start)
                if path.exists() and not overwrite:
                    raise FileExistsError(f"output part already exists: {path}")

    def _path(self, start: int) -> Path:
        end = min(start + self.rows_per_part, self.total_rows)
        return self.output_dir / f"part_{start:08d}_{end:08d}.lance"

    def write(self, rows: Sequence[Dict[str, Any]], first_global_idx: int) -> None:
        cursor = 0
        while cursor < len(rows):
            global_idx = first_global_idx + cursor
            part_start = (global_idx // self.rows_per_part) * self.rows_per_part
            part_end = min(part_start + self.rows_per_part, self.total_rows)
            take = min(len(rows) - cursor, part_end - global_idx)
            chunk = rows[cursor : cursor + take]
            path = self._path(part_start)
            if not self.dry_run:
                table = pa.Table.from_pylist(chunk, schema=OUTPUT_SCHEMA)
                path_key = str(path)
                if path_key not in self.started_parts:
                    mode = "overwrite" if path.exists() and self.overwrite else "create"
                    lance.write_dataset(table, str(path), mode=mode)
                    self.started_parts.add(path_key)
                else:
                    lance.write_dataset(table, str(path), mode="append")
            self.part_counts[str(path)] += take
            cursor += take

    def verify(self) -> None:
        if self.dry_run:
            return
        for raw_path, expected_rows in self.part_counts.items():
            dataset = lance.dataset(raw_path)
            if not dataset.schema.equals(OUTPUT_SCHEMA, check_metadata=True):
                raise AssertionError(
                    f"output schema mismatch for {raw_path}\n"
                    f"actual: {dataset.schema}\nexpected: {OUTPUT_SCHEMA}"
                )
            if dataset.count_rows() != expected_rows:
                raise AssertionError(
                    f"row count mismatch for {raw_path}: "
                    f"{dataset.count_rows()} != {expected_rows}"
                )

    @property
    def output_paths(self) -> List[str]:
        return sorted(self.part_counts)


class SingleLanceWriter:
    """Write one independent ``data.lance`` per source dataset."""

    def __init__(self, output_dir: str, total_rows: int, overwrite: bool, dry_run: bool) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "data.lance"
        self.total_rows = total_rows
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.started = False
        self.rows_written = 0
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and not overwrite:
                raise FileExistsError(f"output Lance already exists: {self.path}")

    def write(self, rows: Sequence[Dict[str, Any]], first_global_idx: int) -> None:
        if not rows:
            return
        if first_global_idx != self.rows_written:
            raise AssertionError(
                f"non-contiguous output rows: first={first_global_idx}, expected={self.rows_written}"
            )
        if not self.dry_run:
            table = pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA)
            if not self.started:
                mode = "overwrite" if self.path.exists() and self.overwrite else "create"
                lance.write_dataset(table, str(self.path), mode=mode)
                self.started = True
            else:
                lance.write_dataset(table, str(self.path), mode="append")
        self.rows_written += len(rows)

    def verify(self) -> None:
        if self.rows_written != self.total_rows:
            raise AssertionError(f"wrote {self.rows_written} rows, expected {self.total_rows}")
        if self.dry_run:
            return
        dataset = lance.dataset(str(self.path))
        if not dataset.schema.equals(OUTPUT_SCHEMA, check_metadata=True):
            raise AssertionError(
                f"output schema mismatch for {self.path}\n"
                f"actual: {dataset.schema}\nexpected: {OUTPUT_SCHEMA}"
            )
        if dataset.count_rows() != self.total_rows:
            raise AssertionError(
                f"row count mismatch for {self.path}: "
                f"{dataset.count_rows()} != {self.total_rows}"
            )

    @property
    def output_paths(self) -> List[str]:
        return [str(self.path)] if self.rows_written else []


def _parse_key_value(items: Optional[Sequence[str]], option: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{option} expects DATASET_KEY=PATH, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise ValueError(f"invalid {option} value {item!r}")
        result[key] = value
    return result


def _expanded_path(value: str, base_dir: Optional[Path] = None) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path)


def _manifest_specs(path: str) -> List[InputSpec]:
    manifest_path = Path(_expanded_path(path)).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("datasets") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ValueError("manifest must be a list or an object with a 'datasets' list")

    specs: List[InputSpec] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError(f"manifest datasets[{index}] must be an object")
        if item.get("enabled", True) is False:
            continue
        key = str(item.get("dataset_key", "")).strip()
        lance_path = item.get("lance_path") or item.get("input")
        if not key or lance_path is None:
            raise ValueError(
                f"manifest datasets[{index}] requires dataset_key and lance_path"
            )
        if not re.fullmatch(r"[A-Za-z0-9._-]+", key):
            raise ValueError(f"unsafe dataset_key {key!r}")

        def optional_path(name: str) -> Optional[str]:
            value = item.get(name)
            return None if value in (None, "") else _expanded_path(str(value), manifest_path.parent)

        specs.append(
            InputSpec(
                dataset_key=key,
                lance_path=_expanded_path(str(lance_path), manifest_path.parent),
                catalog_path=optional_path("catalog_path"),
                channel_map_path=optional_path("channel_map_path"),
            )
        )
    return specs


def build_input_specs(args: argparse.Namespace) -> List[InputSpec]:
    inputs = _parse_key_value(args.input, "--input")
    catalogs = _parse_key_value(args.catalog, "--catalog")
    mappings = _parse_key_value(args.channel_map, "--channel-map")
    specs = _manifest_specs(args.manifest) if args.manifest else []
    known = {spec.dataset_key for spec in specs}
    duplicate_cli = known & set(inputs)
    if duplicate_cli:
        raise ValueError(f"dataset keys occur in both manifest and --input: {sorted(duplicate_cli)}")
    for key, path in inputs.items():
        if not re.fullmatch(r"[A-Za-z0-9._-]+", key):
            raise ValueError(f"unsafe dataset_key {key!r}")
        specs.append(InputSpec(key, _expanded_path(path)))
    known = {spec.dataset_key for spec in specs}
    if len(known) != len(specs):
        raise ValueError("dataset_key values must be unique")
    unknown = (set(catalogs) | set(mappings)) - known
    if unknown:
        raise ValueError(f"catalog/channel-map supplied for unknown datasets: {sorted(unknown)}")

    # Explicit CLI catalog/map values override manifest values.
    specs = [
        InputSpec(
            spec.dataset_key,
            spec.lance_path,
            _expanded_path(catalogs[spec.dataset_key])
            if spec.dataset_key in catalogs
            else spec.catalog_path,
            _expanded_path(mappings[spec.dataset_key])
            if spec.dataset_key in mappings
            else spec.channel_map_path,
        )
        for spec in specs
    ]
    selected = set(args.dataset or [])
    if selected:
        missing = selected - {spec.dataset_key for spec in specs}
        if missing:
            raise ValueError(f"--dataset keys not found in inputs/manifest: {sorted(missing)}")
        specs = [spec for spec in specs if spec.dataset_key in selected]
    return specs


def input_columns(schema: pa.Schema) -> List[str]:
    required = {
        "recording_id",
        "start_seconds",
        "valid_samples",
        "sample_rate_hz",
        "channel_count",
        "eeg",
        "channel_mask",
    }
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"input Lance is missing required columns: {sorted(missing)}")
    optional = {"channel_names", "dataset_id", "start_sample", "stop_sample"}
    return [name for name in schema.names if name in required | optional]


def channel_preflight_columns(schema: pa.Schema) -> List[str]:
    required = {"recording_id", "channel_count", "channel_mask"}
    missing = required - set(schema.names)
    if missing:
        raise ValueError(f"input Lance is missing channel audit columns: {sorted(missing)}")
    optional = {"channel_names"}
    return [name for name in schema.names if name in required | optional]


def preflight_channel_mapping(
    dataset: lance.LanceDataset,
    resolver: CatalogResolver,
    overrides: Mapping[str, str],
    batch_size: int,
) -> ChannelPreflight:
    """Scan channel metadata only and reject a dataset before any output is created."""
    result = ChannelPreflight()
    columns = channel_preflight_columns(dataset.schema)
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            names, source = resolver.resolve(row)
            count = int(row.get("channel_count", len(names)))
            mask = np.asarray(row.get("channel_mask", [True] * count), dtype=bool)
            if len(names) != count or mask.shape != (count,):
                raise ValueError(
                    f"channel metadata length mismatch for recording "
                    f"{row.get('recording_id', '')!r}: names={len(names)}, "
                    f"channel_count={count}, mask_shape={mask.shape}"
                )
            result.rows_scanned += 1
            result.catalog_sources[source] += 1
            result.catalog_signatures.add(tuple(names))
            destination_counts: Counter = Counter()
            for name, active in zip(names, mask):
                if not bool(active):
                    continue
                destination = canonical_channel_name(name, overrides)
                if destination is None:
                    result.unmapped_channels[str(name)] += 1
                    continue
                result.source_to_standard.setdefault(str(name), destination)
                destination_counts[destination] += 1
            duplicates = [name for name, count_ in destination_counts.items() if count_ > 1]
            result.duplicate_destinations.update(duplicates)
            mapped_count = len(destination_counts)
            result.valid_channel_counts[mapped_count] += 1
            result.zero_mapped_rows += int(mapped_count == 0)
    return result


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def _preflight_summary(preflight: ChannelPreflight) -> Dict[str, Any]:
    return {
        "eligible": preflight.eligible,
        "unmapped_channels_are_dropped": True,
        "rows_scanned": preflight.rows_scanned,
        "catalog_sources": _counter_dict(preflight.catalog_sources),
        "unique_channel_catalogs": len(preflight.catalog_signatures),
        "source_to_standard": dict(sorted(preflight.source_to_standard.items())),
        "unmapped_channels": _counter_dict(preflight.unmapped_channels),
        "duplicate_destinations": _counter_dict(preflight.duplicate_destinations),
        "mapped_channel_count_distribution": _counter_dict(preflight.valid_channel_counts),
        "zero_mapped_rows": preflight.zero_mapped_rows,
    }


def run(args: argparse.Namespace) -> None:
    specs = build_input_specs(args)
    if not specs:
        raise ValueError("provide at least one --input or a --manifest")

    output_root = Path(args.output_root)
    summary_path = output_root / "conversion_summary.json"
    if not args.dry_run and summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"summary already exists: {summary_path}; use a new --output-root"
        )
    summaries: List[Dict[str, Any]] = []
    opened: List[
        Tuple[InputSpec, lance.LanceDataset, CatalogResolver, Dict[str, str], ChannelPreflight]
    ] = []

    # Audit every row's channel metadata before creating any dataset output.
    for spec in specs:
        dataset = lance.dataset(spec.lance_path)
        resolver = CatalogResolver(dataset.schema, spec.catalog_path)
        overrides = load_channel_override(spec.channel_map_path)
        input_columns(dataset.schema)
        preflight = preflight_channel_mapping(
            dataset, resolver, overrides, batch_size=args.read_batch_size
        )
        opened.append((spec, dataset, resolver, overrides, preflight))
        print(
            f"channel_preflight dataset={spec.dataset_key} rows={dataset.count_rows()} "
            f"eligible={preflight.eligible} "
            f"dropped_unmapped={sorted(preflight.unmapped_channels)} "
            f"duplicates={sorted(preflight.duplicate_destinations)}",
            flush=True,
        )

    for spec, dataset, resolver, overrides, preflight in opened:
        source_rows = dataset.count_rows()
        total_rows = min(source_rows, args.limit) if args.limit is not None else source_rows
        mapped_standard = sorted(set(preflight.source_to_standard.values()))
        base_summary: Dict[str, Any] = {
            "dataset_key": spec.dataset_key,
            "input_lance": spec.lance_path,
            "catalog_path": spec.catalog_path,
            "channel_map_path": spec.channel_map_path,
            "source_rows": source_rows,
            "requested_rows": total_rows,
            "channel_preflight": _preflight_summary(preflight),
            "mapped_standard_channels": mapped_standard,
            "missing_standard_channels": [
                name for name in STANDARD_64_CHANNELS if name not in set(mapped_standard)
            ],
        }
        if not preflight.eligible:
            base_summary.update(
                {
                    "status": "SKIPPED_CHANNEL_MISMATCH",
                    "processed_rows": 0,
                    "output_paths": [],
                }
            )
            summaries.append(base_summary)
            print(
                f"SKIP dataset={spec.dataset_key}: no usable mapping in one or more rows "
                f"or duplicate destination mapping",
                flush=True,
            )
            continue

        dataset_output_dir = output_root / spec.dataset_key
        if args.layout == "single":
            writer: Any = SingleLanceWriter(
                str(dataset_output_dir), total_rows, args.overwrite, args.dry_run
            )
        else:
            writer = PartWriter(
                str(dataset_output_dir),
                total_rows=total_rows,
                rows_per_part=args.rows_per_part,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

        stats = ConversionStats()
        local_idx = 0
        columns = input_columns(dataset.schema)
        scanner = dataset.scanner(columns=columns, batch_size=args.read_batch_size)
        for record_batch in scanner.to_batches():
            if local_idx >= total_rows:
                break
            raw_rows = record_batch.to_pylist()
            remaining = total_rows - local_idx
            if len(raw_rows) > remaining:
                raw_rows = raw_rows[:remaining]
            converted = [
                convert_row(
                    row,
                    dataset_key=spec.dataset_key,
                    global_idx=local_idx + offset,
                    resolver=resolver,
                    overrides=overrides,
                    stats=stats,
                    allow_unmapped_channels=True,
                )
                for offset, row in enumerate(raw_rows)
            ]
            writer.write(converted, first_global_idx=local_idx)
            local_idx += len(converted)
            print(
                f"dataset={spec.dataset_key} converted={local_idx}/{total_rows}",
                flush=True,
            )

        if local_idx != total_rows:
            raise AssertionError(
                f"dataset={spec.dataset_key}: converted {local_idx}, expected {total_rows}"
            )
        writer.verify()
        per_row_ms = 1000.0 * stats.processing_seconds / max(stats.rows, 1)
        base_summary.update(
            {
                "status": "DRY_RUN_PASSED" if args.dry_run else "CONVERTED",
                "processed_rows": stats.rows,
                "tail_rows": stats.tail_rows,
                "input_valid_samples": _counter_dict(stats.input_valid_samples),
                "output_valid_samples": _counter_dict(stats.output_valid_samples),
                "valid_channel_counts": _counter_dict(stats.valid_channel_counts),
                "dropped_channels": _counter_dict(stats.dropped_channels),
                "duplicate_destinations": _counter_dict(stats.duplicate_destinations),
                "processing_ms_per_row": per_row_ms,
                "output_paths": writer.output_paths,
            }
        )
        summaries.append(base_summary)

    report = {
        "format": "pretrain_V2_to_FBMAE_summary_v1",
        "dry_run": bool(args.dry_run),
        "layout": args.layout,
        "channel_policy": "subset-match; >=1 mapped active channel per row; drop unmapped",
        "strict_channel_matching": False,
        "output_root": str(output_root),
        "datasets": summaries,
    }
    print("\n===== Dataset conversion overview =====")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"summary_json: {summary_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        metavar="DATASET_KEY=PATH",
        help="mounted input Lance; repeat for multiple independently written datasets",
    )
    parser.add_argument(
        "--manifest",
        help="JSON dataset list; paths may contain environment variables",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="only process this dataset_key from the manifest/input list; repeat as needed",
    )
    parser.add_argument(
        "--catalog",
        action="append",
        metavar="DATASET_KEY=JSON",
        help="ordered channel catalog or canonical preflight JSON",
    )
    parser.add_argument(
        "--channel-map",
        action="append",
        metavar="DATASET_KEY=JSON",
        help="authoritative source-name to standard-name map; required for generic labels",
    )
    parser.add_argument(
        "--output-root",
        "--output-dir",
        dest="output_root",
        required=True,
        help="root directory; each dataset is written to its own subdirectory",
    )
    parser.add_argument(
        "--layout",
        choices=("single", "parts"),
        default="single",
        help="single writes DATASET/data.lance (default); parts keeps part_*.lance compatibility",
    )
    parser.add_argument("--rows-per-part", type=int, default=10_000)
    parser.add_argument("--read-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true", help="process and validate without writing Lance")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite only the exact generated dataset outputs and summary",
    )
    args = parser.parse_args(argv)
    if args.rows_per_part <= 0 or args.read_batch_size <= 0:
        parser.error("--rows-per-part and --read-batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
