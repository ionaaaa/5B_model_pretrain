"""Runtime adaptation for the canonical Inria EEG Lance release.

This module intentionally does not write a processed Lance dataset.  It reads
100 Hz, 10 second, per-channel standardized canonical EEG rows and constructs
the tensors consumed by FBMAE inside the DataLoader worker.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lance
import numpy as np
import torch
from scipy.signal import spectrogram
from torch.utils.data import Dataset

from channel_config import STANDARD_64_CHANNELS
from configs import Config


INRIA_CHANNELS: Tuple[str, ...] = (
    "Fz", "FCz", "Cz", "CPz", "Pz",
    "C1", "C3", "C5", "C2", "C4", "C6",
    "F4", "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4",
    "F3", "FC1", "FC3", "FC5", "CP1", "CP3", "CP5", "P3",
)

SOURCE_COLUMNS: Tuple[str, ...] = (
    "eeg",
    "eeg_shape",
    "eeg_n_channels",
    "eeg_channel_indices",
    "valid_channel_mask_json",
    "valid_samples",
    "source_index",
    "subject",
    "dataset_id",
)


def load_storage_options(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    if "storage_options" in values and isinstance(values["storage_options"], dict):
        values = values["storage_options"]
    return {str(key): str(value) for key, value in values.items()}


def _as_python(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def _parse_mask_payload(value: Any, channel_names: Sequence[str]) -> np.ndarray:
    """Parse a row-level valid channel mask in row channel order."""
    value = _as_python(value)
    if value is None or value == "":
        return np.ones(len(channel_names), dtype=bool)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return np.ones(len(channel_names), dtype=bool)

    if isinstance(value, dict):
        for key in ("valid_channel_mask", "channel_mask", "mask", "values"):
            if key in value:
                return _parse_mask_payload(value[key], channel_names)
        if "valid_channels" in value:
            valid_names = {str(name) for name in value["valid_channels"]}
            return np.asarray([name in valid_names for name in channel_names], dtype=bool)
        if all(name in value for name in channel_names):
            return np.asarray([bool(value[name]) for name in channel_names], dtype=bool)
        raise ValueError(
            "Unsupported valid_channel_mask_json object. "
            f"keys={sorted(map(str, value.keys()))[:20]}"
        )

    mask = np.asarray(value).reshape(-1)
    if mask.size != len(channel_names):
        raise ValueError(
            "valid channel mask length mismatch: "
            f"{mask.size} != {len(channel_names)}"
        )
    return mask.astype(bool)


def _row_channel_names(row: Dict[str, Any], n_channels: int) -> Tuple[List[str], np.ndarray]:
    indices_value = _as_python(row.get("eeg_channel_indices"))
    if indices_value is None:
        indices = np.arange(n_channels, dtype=np.int64)
    else:
        indices = np.asarray(indices_value, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            indices = np.arange(n_channels, dtype=np.int64)

    if indices.size != n_channels:
        raise ValueError(
            f"eeg_channel_indices length={indices.size}, expected {n_channels}"
        )
    if indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(INRIA_CHANNELS):
        raise ValueError(
            "Inria eeg_channel_indices must be zero-based catalog indices in "
            f"[0, {len(INRIA_CHANNELS) - 1}], got range "
            f"[{indices.min()}, {indices.max()}]"
        )
    return [INRIA_CHANNELS[int(i)] for i in indices], indices


def _reshape_eeg(row: Dict[str, Any]) -> np.ndarray:
    flat = np.asarray(_as_python(row["eeg"]), dtype=np.float32).reshape(-1)
    shape = np.asarray(_as_python(row["eeg_shape"]), dtype=np.int64).reshape(-1)
    n_channels = int(_as_python(row["eeg_n_channels"]))

    if shape.size == 2 and int(np.prod(shape)) == flat.size:
        eeg = flat.reshape(tuple(int(x) for x in shape))
        if eeg.shape[0] == n_channels:
            return eeg.astype(np.float32, copy=False)
        if eeg.shape[1] == n_channels:
            return eeg.T.astype(np.float32, copy=False)
    if n_channels > 0 and flat.size % n_channels == 0:
        return flat.reshape(n_channels, flat.size // n_channels).astype(np.float32)
    raise ValueError(
        f"Cannot recover [channel,time] EEG: eeg_shape={shape.tolist()}, "
        f"eeg_n_channels={n_channels}, flat_size={flat.size}"
    )


def align_inria_to_standard64(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    row_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(channel_names) != eeg.shape[0]:
        raise ValueError("channel name count does not match EEG channel axis")
    if row_valid_mask.shape != (eeg.shape[0],):
        raise ValueError("row valid channel mask does not match EEG channel axis")

    standard_index = {name: index for index, name in enumerate(STANDARD_64_CHANNELS)}
    signal64 = np.zeros((len(STANDARD_64_CHANNELS), eeg.shape[1]), dtype=np.float32)
    channel_valid_mask = np.zeros(len(STANDARD_64_CHANNELS), dtype=np.float32)

    for source_index, name in enumerate(channel_names):
        if name not in standard_index:
            raise ValueError(f"Inria channel {name!r} is absent from STANDARD_64_CHANNELS")
        destination = standard_index[name]
        if channel_valid_mask[destination] != 0:
            raise ValueError(f"Duplicate Inria mapping to standard channel {name!r}")
        if row_valid_mask[source_index]:
            signal64[destination] = eeg[source_index]
            channel_valid_mask[destination] = 1.0

    return signal64, channel_valid_mask


def _next_power_of_two(value: int) -> int:
    return 1 << int(math.ceil(math.log2(value)))


def compute_runtime_targets(
    signal64: np.ndarray,
    channel_valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the existing offline FBMAE time-frequency target pipeline."""
    cfg = Config().data
    sfreq = float(cfg.target_sfreq)
    nperseg = int(round(cfg.stft_window_seconds * sfreq))
    hop = int(round(cfg.stft_hop_seconds * sfreq))
    noverlap = max(0, nperseg - hop)
    nfft = _next_power_of_two(nperseg)

    frequencies, stft_times, psd = spectrogram(
        signal64,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="density",
        mode="psd",
        axis=1,
    )
    stft_times = stft_times.astype(np.float32)

    band_trajectories = []
    for _, low, high in cfg.band_defs:
        frequency_mask = (frequencies >= low) & (frequencies < high)
        if not frequency_mask.any():
            band = np.zeros((signal64.shape[0], psd.shape[-1]), dtype=np.float32)
        else:
            band = psd[:, frequency_mask, :].mean(axis=1)
        band_trajectories.append(np.log(band + cfg.eps).astype(np.float32))

    band_power = np.transpose(np.stack(band_trajectories, axis=0), (1, 0, 2))
    continuous_times = np.arange(signal64.shape[1], dtype=np.float32) / sfreq
    timefreq = np.zeros(
        (signal64.shape[0], len(cfg.band_defs), signal64.shape[1]),
        dtype=np.float32,
    )
    for channel in range(signal64.shape[0]):
        for band in range(len(cfg.band_defs)):
            timefreq[channel, band] = np.interp(
                continuous_times,
                stft_times,
                band_power[channel, band],
                left=band_power[channel, band, 0],
                right=band_power[channel, band, -1],
            )

    standardized = np.zeros_like(timefreq, dtype=np.float32)
    valid = channel_valid_mask.astype(bool)
    if valid.any():
        values = timefreq[valid]
        means = values.mean(axis=2, keepdims=True)
        stds = values.std(axis=2, keepdims=True)
        standardized[valid] = (values - means) / (stds + 1e-8)

    patch_len = int(round(cfg.patch_seconds * sfreq))
    patch_stride = int(round(cfg.patch_stride_seconds * sfreq))
    n_patches = 1 + (signal64.shape[1] - patch_len) // patch_stride
    frames_per_patch = int(round(cfg.patch_seconds / cfg.target_frame_seconds))
    frame_offsets = np.linspace(0, patch_len - 1, frames_per_patch).astype(np.int64)

    target4d = np.zeros(
        (signal64.shape[0], n_patches, len(cfg.band_defs), frames_per_patch),
        dtype=np.float32,
    )
    for patch_index in range(n_patches):
        indices = patch_index * patch_stride + frame_offsets
        target4d[:, patch_index] = standardized[:, :, indices]
    return target4d.reshape(-1, len(cfg.band_defs), frames_per_patch), target4d


def adapt_inria_row(row: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
    valid_samples = int(_as_python(row["valid_samples"]))
    if valid_samples != 1000:
        raise ValueError(f"Only valid_samples == 1000 is supported, got {valid_samples}")

    adaptation_started = time.perf_counter()
    eeg = _reshape_eeg(row)
    if eeg.shape[1] != 1000:
        raise ValueError(f"Expected 1000 time samples, got EEG shape {eeg.shape}")
    channel_names, channel_indices = _row_channel_names(row, eeg.shape[0])

    raw_mask_value = _as_python(row.get("valid_channel_mask_json"))
    if isinstance(raw_mask_value, str):
        decoded = json.loads(raw_mask_value) if raw_mask_value else None
    else:
        decoded = raw_mask_value

    # A catalog-sized mask must be gathered into the row's selected channel order.
    if isinstance(decoded, list) and len(decoded) == len(INRIA_CHANNELS):
        row_valid_mask = np.asarray(decoded, dtype=bool)[channel_indices]
    else:
        row_valid_mask = _parse_mask_payload(decoded, channel_names)

    signal64, channel_valid_mask = align_inria_to_standard64(
        eeg=eeg,
        channel_names=channel_names,
        row_valid_mask=row_valid_mask,
    )

    cfg = Config().data
    patch_len = int(round(cfg.patch_seconds * cfg.target_sfreq))
    patch_stride = int(round(cfg.patch_stride_seconds * cfg.target_sfreq))
    n_patches = 1 + (signal64.shape[1] - patch_len) // patch_stride
    patch4d = np.stack(
        [
            signal64[:, index * patch_stride:index * patch_stride + patch_len]
            for index in range(n_patches)
        ],
        axis=1,
    ).astype(np.float32)
    token_inputs = patch4d.reshape(-1, patch_len)
    token_channel_indices = np.repeat(np.arange(64), n_patches).astype(np.int64)
    token_time_indices = np.tile(np.arange(n_patches), 64).astype(np.int64)
    token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)
    adaptation_ms = (time.perf_counter() - adaptation_started) * 1000.0

    target_started = time.perf_counter()
    targets, target4d = compute_runtime_targets(signal64, channel_valid_mask)
    target_ms = (time.perf_counter() - target_started) * 1000.0

    if validate:
        tensors = {
            "token_inputs": token_inputs,
            "targets": targets,
            "token_channel_indices": token_channel_indices,
            "token_time_indices": token_time_indices,
            "token_valid_mask": token_valid_mask,
            "channel_valid_mask": channel_valid_mask,
        }
        for name, tensor in tensors.items():
            if not np.isfinite(tensor).all():
                raise AssertionError(f"{name} contains NaN or Inf")

        if token_inputs.shape != (640, 100) or targets.shape != (640, 5, 5):
            raise AssertionError(
                f"Unexpected token/target shapes: {token_inputs.shape}, {targets.shape}"
            )
        expected_channels = np.repeat(np.arange(64), 10)
        expected_times = np.tile(np.arange(10), 64)
        if not np.array_equal(token_channel_indices, expected_channels):
            raise AssertionError("token ordering is not channel-major")
        if not np.array_equal(token_time_indices, expected_times):
            raise AssertionError("token time ordering is not channel-major")
        if token_channel_indices.min() < 0 or token_channel_indices.max() >= 64:
            raise AssertionError("token channel index out of range")
        if token_time_indices.min() < 0 or token_time_indices.max() >= 10:
            raise AssertionError("token time index out of range")
        if not np.array_equal(token_inputs.reshape(64, 10, 100), patch4d):
            raise AssertionError("token_inputs flatten order does not match waveform patches")
        if not np.array_equal(targets.reshape(64, 10, 5, 5), target4d):
            raise AssertionError("target flatten order does not match token_inputs order")

        invalid = channel_valid_mask == 0
        if np.any(signal64[invalid] != 0) or np.any(patch4d[invalid] != 0):
            raise AssertionError("invalid channel waveform/token data is not zero")
        if np.any(target4d[invalid] != 0):
            raise AssertionError("invalid channel targets are not zero")
        valid_targets = target4d[channel_valid_mask.astype(bool)]
        if valid_targets.size == 0 or float(valid_targets.std()) <= 1e-6:
            raise AssertionError("targets are constant or no valid targets exist")

    valid_waveform = signal64[channel_valid_mask.astype(bool)]
    waveform_mean_abs_max = float(np.max(np.abs(valid_waveform.mean(axis=1))))
    waveform_std_error_max = float(np.max(np.abs(valid_waveform.std(axis=1) - 1.0)))

    return {
        "token_inputs": torch.from_numpy(token_inputs),
        "targets": torch.from_numpy(targets),
        "token_channel_indices": torch.from_numpy(token_channel_indices),
        "token_time_indices": torch.from_numpy(token_time_indices),
        "token_valid_mask": torch.from_numpy(token_valid_mask),
        "channel_valid_mask": torch.from_numpy(channel_valid_mask),
        "adaptation_time_ms": adaptation_ms,
        "target_time_ms": target_ms,
        "valid_channel_count": int(channel_valid_mask.sum()),
        "waveform_mean_abs_max": waveform_mean_abs_max,
        "waveform_std_error_max": waveform_std_error_max,
        "source_index": int(_as_python(row.get("source_index", -1))),
        "subject": str(_as_python(row.get("subject", "")) or ""),
        "dataset_id": str(_as_python(row.get("dataset_id", "inria")) or "inria"),
    }


class InriaRuntimeTargetDataset(Dataset):
    """Map-style source Lance dataset with worker-side FBMAE target generation."""

    def __init__(
        self,
        lance_uri: str,
        storage_options: Optional[Dict[str, str]] = None,
        max_samples: Optional[int] = None,
        validate: bool = True,
        index_scan_batch_size: int = 65536,
    ):
        self.lance_uri = str(lance_uri)
        self.storage_options = storage_options
        self.validate = bool(validate)
        self.ds = None

        dataset = self._open_new()
        missing = sorted(set(SOURCE_COLUMNS) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"Source Lance is missing required columns: {missing}")
        self.source_indices = self._find_full_length_rows(
            dataset,
            max_samples=max_samples,
            batch_size=index_scan_batch_size,
        )
        del dataset
        if not self.source_indices:
            raise ValueError("No Inria rows with valid_samples == 1000 were found")

    def _open_new(self):
        if self.storage_options is None:
            return lance.dataset(self.lance_uri)
        return lance.dataset(self.lance_uri, storage_options=self.storage_options)

    @staticmethod
    def _find_full_length_rows(dataset, max_samples: Optional[int], batch_size: int) -> List[int]:
        selected: List[int] = []
        row_offset = 0
        scanner = dataset.scanner(columns=["valid_samples"], batch_size=batch_size)
        for batch in scanner.to_batches():
            values = np.asarray(batch.column(0).to_pylist(), dtype=object)
            local_indices = np.flatnonzero(values == 1000)
            selected.extend((row_offset + local_indices).astype(np.int64).tolist())
            row_offset += batch.num_rows
            if max_samples is not None and len(selected) >= max_samples:
                return selected[:max_samples]
        return selected

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ds"] = None
        return state

    def _open(self):
        if self.ds is None:
            self.ds = self._open_new()
        return self.ds

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source_index = self.source_indices[int(index)]
        row = self._open().take([source_index], columns=list(SOURCE_COLUMNS)).to_pylist()[0]
        return adapt_inria_row(row, validate=self.validate)

    def __getitems__(self, indices: Iterable[int]) -> List[Dict[str, Any]]:
        requested = [int(index) for index in indices]
        if not requested:
            return []
        indexed_sources = [self.source_indices[index] for index in requested]
        order = np.argsort(indexed_sources)
        sorted_sources = [indexed_sources[int(i)] for i in order]
        rows = self._open().take(
            sorted_sources,
            columns=list(SOURCE_COLUMNS),
        ).to_pylist()
        sorted_samples = [adapt_inria_row(row, validate=self.validate) for row in rows]
        restored: List[Optional[Dict[str, Any]]] = [None] * len(sorted_samples)
        for sorted_position, original_position in enumerate(order):
            restored[int(original_position)] = sorted_samples[sorted_position]
        return [sample for sample in restored if sample is not None]
