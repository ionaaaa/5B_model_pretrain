"""Metadata-driven decoder for canonical EEG Lance rows.

The canonical releases share one physical schema. Dataset-specific channel
semantics live in ``omni_neuralset.segment_lance`` schema metadata.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from runtime_adaptation.contracts import CanonicalEEGWindow


SCHEMA_METADATA_KEY = "omni_neuralset.segment_lance"
CANONICAL_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "eeg", "eeg_shape", "eeg_n_channels", "eeg_channel_indices",
    "valid_channel_mask_json", "valid_samples", "source_index", "subject",
    "dataset_id",
)
CANONICAL_OPTIONAL_COLUMNS: Tuple[str, ...] = ("segment_id", "eeg_channel_map_id")


def as_python(value: Any) -> Any:
    return value.as_py() if hasattr(value, "as_py") else value


def normalize_integer_scalar(value: Any) -> Optional[int]:
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
            f"valid channel mask length mismatch: {mask.size} != {len(channel_names)}"
        )
    return mask.astype(bool)


def parse_canonical_schema_metadata(schema_or_metadata: Any) -> Dict[str, Any]:
    """Decode and validate the canonical JSON schema-metadata envelope."""
    metadata = getattr(schema_or_metadata, "metadata", schema_or_metadata)
    if not metadata:
        raise ValueError("Canonical Lance schema has no schema metadata")
    raw = metadata.get(SCHEMA_METADATA_KEY)
    if raw is None:
        raw = metadata.get(SCHEMA_METADATA_KEY.encode("utf-8"))
    if raw is None:
        raise ValueError(
            f"Canonical Lance schema metadata lacks {SCHEMA_METADATA_KEY!r}"
        )
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Canonical Lance schema metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Canonical Lance schema metadata payload must be an object")
    return raw


class MetadataDrivenCanonicalDecoder:
    """Decode the common physical schema using its embedded channel catalog."""

    source_columns = CANONICAL_REQUIRED_COLUMNS + CANONICAL_OPTIONAL_COLUMNS

    def __init__(
        self,
        *,
        dataset_key: str,
        sfreq: float,
        channel_catalogs: Mapping[str, Sequence[str]],
        default_channel_map_id: Optional[str] = None,
        schema_metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.dataset_key = str(dataset_key)
        self.source_sfreq = float(sfreq)
        self.channel_catalogs = {
            str(map_id): tuple(str(name) for name in names)
            for map_id, names in channel_catalogs.items()
        }
        if not self.dataset_key:
            raise ValueError("dataset_key from schema metadata is empty")
        if not np.isfinite(self.source_sfreq) or self.source_sfreq <= 0:
            raise ValueError(f"Invalid canonical sampling rate: {self.source_sfreq}")
        if not self.channel_catalogs:
            raise ValueError("No EEG channel catalogs found in schema metadata")
        if default_channel_map_id is None and len(self.channel_catalogs) == 1:
            default_channel_map_id = next(iter(self.channel_catalogs))
        self.default_channel_map_id = (
            None if default_channel_map_id is None else str(default_channel_map_id)
        )
        if (
            self.default_channel_map_id is not None
            and self.default_channel_map_id not in self.channel_catalogs
        ):
            raise ValueError(
                f"Unknown default channel_map_id={self.default_channel_map_id!r}"
            )
        self.schema_metadata = dict(schema_metadata or {})

    @classmethod
    def from_schema(
        cls,
        schema: Any,
        *,
        expected_dataset_key: Optional[str] = None,
    ) -> "MetadataDrivenCanonicalDecoder":
        payload = parse_canonical_schema_metadata(schema)
        dataset_key = str(payload.get("dataset", {}).get("dataset_id", ""))
        if expected_dataset_key and dataset_key != expected_dataset_key:
            raise ValueError(
                f"Dataset identity mismatch: requested {expected_dataset_key!r}, "
                f"schema metadata says {dataset_key!r}"
            )
        try:
            sfreq = float(payload["recipe"]["sampling_rates"]["eeg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Canonical output sampling rate is absent from "
                "recipe.sampling_rates.eeg"
            ) from exc

        entries = payload.get("channel_metadata_catalog", {}).get("eeg", [])
        catalogs: Dict[str, Tuple[str, ...]] = {}
        for entry in entries:
            map_id = str(entry.get("channel_map_id", ""))
            names = entry.get("channel_names")
            if not map_id or not isinstance(names, list) or not names:
                raise ValueError("Malformed EEG channel catalog in schema metadata")
            if map_id in catalogs:
                raise ValueError(f"Duplicate EEG channel_map_id={map_id!r}")
            catalogs[map_id] = tuple(str(name) for name in names)
        return cls(
            dataset_key=dataset_key,
            sfreq=sfreq,
            channel_catalogs=catalogs,
            schema_metadata=payload,
        )

    @staticmethod
    def _reshape_eeg(row: Mapping[str, Any]) -> np.ndarray:
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

    def _resolve_catalog(self, row: Mapping[str, Any]) -> Tuple[str, Tuple[str, ...]]:
        map_id = str(as_python(row.get("eeg_channel_map_id")) or "")
        if not map_id:
            map_id = self.default_channel_map_id or ""
        if map_id not in self.channel_catalogs:
            raise ValueError(
                f"Row channel_map_id={map_id!r} is not present in schema metadata; "
                f"available={sorted(self.channel_catalogs)}"
            )
        return map_id, self.channel_catalogs[map_id]

    def decode(self, row: Mapping[str, Any]) -> CanonicalEEGWindow:
        valid_samples = normalize_integer_scalar(row.get("valid_samples"))
        if valid_samples is None:
            raise ValueError(f"Invalid valid_samples: {row.get('valid_samples')!r}")
        row_dataset = str(as_python(row.get("dataset_id")) or self.dataset_key)
        if row_dataset != self.dataset_key:
            raise ValueError(
                f"Row dataset_id={row_dataset!r} disagrees with schema "
                f"dataset_id={self.dataset_key!r}"
            )

        eeg = self._reshape_eeg(row)
        map_id, catalog = self._resolve_catalog(row)
        raw_indices = as_python(row.get("eeg_channel_indices"))
        indices = (
            np.arange(eeg.shape[0], dtype=np.int64)
            if raw_indices is None
            else np.asarray(raw_indices, dtype=np.int64).reshape(-1)
        )
        if indices.size == 0:
            indices = np.arange(eeg.shape[0], dtype=np.int64)
        if indices.size != eeg.shape[0]:
            raise ValueError(
                f"eeg_channel_indices length={indices.size}, expected {eeg.shape[0]}"
            )
        if indices.min(initial=0) < 0 or indices.max(initial=-1) >= len(catalog):
            raise ValueError(
                f"eeg_channel_indices must be in [0, {len(catalog) - 1}], got "
                f"[{indices.min()}, {indices.max()}]"
            )
        channel_names = tuple(catalog[int(index)] for index in indices)

        raw_mask = as_python(row.get("valid_channel_mask_json"))
        decoded_mask = json.loads(raw_mask) if isinstance(raw_mask, str) and raw_mask else raw_mask
        if isinstance(decoded_mask, list) and len(decoded_mask) == len(catalog):
            row_valid = np.asarray(decoded_mask, dtype=bool)[indices]
        else:
            row_valid = parse_mask_payload(decoded_mask, channel_names)

        source_index = normalize_integer_scalar(row.get("source_index"))
        source_index = -1 if source_index is None else source_index
        segment_id = str(as_python(row.get("segment_id")) or "")
        return CanonicalEEGWindow(
            waveform=eeg,
            channel_names=channel_names,
            channel_valid=row_valid,
            valid_samples=valid_samples,
            sfreq=self.source_sfreq,
            dataset_key=self.dataset_key,
            sample_id=segment_id or str(source_index),
            subject_id=str(as_python(row.get("subject")) or ""),
            source_index=source_index,
            metadata={
                "channel_map_id": map_id,
                "eeg_channel_indices": indices.copy(),
                "channel_catalog": catalog,
            },
        )
