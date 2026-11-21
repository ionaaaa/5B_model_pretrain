"""Inria source-row decoder for the canonical runtime contract."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from runtime_adaptation.contracts import CanonicalEEGWindow
from runtime_adaptation.canonical import MetadataDrivenCanonicalDecoder


INRIA_CHANNELS: Tuple[str, ...] = (
    "Fz", "FCz", "Cz", "CPz", "Pz",
    "C1", "C3", "C5", "C2", "C4", "C6",
    "F4", "FC2", "FC4", "FC6", "CP2", "CP4", "CP6", "P4",
    "F3", "FC1", "FC3", "FC5", "CP1", "CP3", "CP5", "P3",
)
INRIA_CHANNEL_MAP_ID = "cf8399da04dbb6bb"

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


class InriaDecoder(MetadataDrivenCanonicalDecoder):
    """Decode one Inria canonical Lance row without FBMAE-specific processing."""

    channel_catalog = INRIA_CHANNELS
    source_columns = INRIA_SOURCE_COLUMNS

    def __init__(self):
        super().__init__(
            dataset_key="inria",
            sfreq=100.0,
            channel_catalogs={INRIA_CHANNEL_MAP_ID: INRIA_CHANNELS},
            default_channel_map_id=INRIA_CHANNEL_MAP_ID,
        )
