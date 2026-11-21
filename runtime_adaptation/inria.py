"""Inria source-row decoder for the canonical runtime contract."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from runtime_adaptation.contracts import CanonicalEEGWindow


INRIA_CHANNELS: Tuple[str, ...] = (
    "Fz", "FCz", "Cz", "CPz", "Pz",
    "C1", "C3", "C5", "C2", "C4", "C6",
    "F4", "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4",
    "F3", "FC1", "FC3", "FC5", "CP1", "CP3", "CP5", "P3",
)

INRIA_SOURCE_COLUMNS: Tuple[str, ...] = (
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


def as_python(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def normalize_integer_scalar(value: Any) -> Optional[int]:
    """Normalize Arrow/Python integer-like scalars without changing meaning."""
    value = as_python(value)
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def parse_mask_payload(value: Any, channel_names: Sequence[str]) -> np.ndarray:
    """Parse a row-level valid channel mask in row channel order."""
    value = as_python(value)
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
                return parse_mask_payload(value[key], channel_names)
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


class InriaDecoder:
    """Decode one Inria canonical Lance row without FBMAE-specific processing."""

    dataset_key = "inria"
    source_sfreq = 100.0
    channel_catalog = INRIA_CHANNELS
    source_columns = INRIA_SOURCE_COLUMNS

    @staticmethod
    def _reshape_eeg(row: Dict[str, Any]) -> np.ndarray:
        flat = np.asarray(as_python(row["eeg"]), dtype=np.float32).reshape(-1)
        shape = np.asarray(as_python(row["eeg_shape"]), dtype=np.int64).reshape(-1)
        n_channels = int(as_python(row["eeg_n_channels"]))

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

    def _row_channel_names(
        self,
        row: Dict[str, Any],
        n_channels: int,
    ) -> Tuple[List[str], np.ndarray]:
        indices_value = as_python(row.get("eeg_channel_indices"))
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
        if indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(self.channel_catalog):
            raise ValueError(
                "Inria eeg_channel_indices must be zero-based catalog indices in "
                f"[0, {len(self.channel_catalog) - 1}], got range "
                f"[{indices.min()}, {indices.max()}]"
            )
        return [self.channel_catalog[int(i)] for i in indices], indices

    def decode(self, row: Dict[str, Any]) -> CanonicalEEGWindow:
        valid_samples = normalize_integer_scalar(row.get("valid_samples"))
        if valid_samples is None:
            raise ValueError(f"Invalid Inria valid_samples: {row.get('valid_samples')!r}")

        eeg = self._reshape_eeg(row)
        channel_names, channel_indices = self._row_channel_names(row, eeg.shape[0])
        raw_mask = as_python(row.get("valid_channel_mask_json"))
        if isinstance(raw_mask, str):
            decoded_mask = json.loads(raw_mask) if raw_mask else None
        else:
            decoded_mask = raw_mask

        if isinstance(decoded_mask, list) and len(decoded_mask) == len(self.channel_catalog):
            row_valid = np.asarray(decoded_mask, dtype=bool)[channel_indices]
        else:
            row_valid = parse_mask_payload(decoded_mask, channel_names)

        source_index = normalize_integer_scalar(row.get("source_index"))
        source_index = -1 if source_index is None else source_index
        dataset_key = str(as_python(row.get("dataset_id", self.dataset_key)) or self.dataset_key)
        subject_id = str(as_python(row.get("subject", "")) or "")
        return CanonicalEEGWindow(
            waveform=eeg,
            channel_names=tuple(channel_names),
            channel_valid=row_valid,
            valid_samples=valid_samples,
            sfreq=self.source_sfreq,
            dataset_key=dataset_key,
            sample_id=str(source_index),
            subject_id=subject_id,
            source_index=source_index,
            metadata={"eeg_channel_indices": channel_indices.copy()},
        )
