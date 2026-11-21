"""Dataset-independent contracts used by runtime EEG adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

import numpy as np


@dataclass
class CanonicalEEGWindow:
    """Decoded source window before standard-64 FBMAE adaptation."""

    waveform: np.ndarray
    channel_names: Tuple[str, ...]
    channel_valid: np.ndarray
    valid_samples: int
    sfreq: float
    dataset_key: str
    sample_id: str = ""
    subject_id: str = ""
    source_index: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.waveform = np.asarray(self.waveform, dtype=np.float32)
        self.channel_names = tuple(str(name) for name in self.channel_names)
        self.channel_valid = np.asarray(self.channel_valid, dtype=bool).reshape(-1)
        self.valid_samples = int(self.valid_samples)
        self.sfreq = float(self.sfreq)
        self.dataset_key = str(self.dataset_key)
        self.sample_id = str(self.sample_id)
        self.subject_id = str(self.subject_id)
        self.source_index = int(self.source_index)

        if self.waveform.ndim != 2:
            raise ValueError(
                f"Canonical waveform must be [channel,time], got {self.waveform.shape}"
            )
        if len(self.channel_names) != self.waveform.shape[0]:
            raise ValueError(
                "channel_names length does not match waveform channel axis: "
                f"{len(self.channel_names)} != {self.waveform.shape[0]}"
            )
        if self.channel_valid.shape != (self.waveform.shape[0],):
            raise ValueError(
                "channel_valid does not match waveform channel axis: "
                f"{self.channel_valid.shape} != {(self.waveform.shape[0],)}"
            )
        if not 0 <= self.valid_samples <= self.waveform.shape[1]:
            raise ValueError(
                f"valid_samples={self.valid_samples} is outside "
                f"[0, {self.waveform.shape[1]}]"
            )
        if not np.isfinite(self.sfreq) or self.sfreq <= 0:
            raise ValueError(f"sfreq must be finite and positive, got {self.sfreq}")
        if not self.dataset_key:
            raise ValueError("dataset_key must be non-empty")
        if not np.isfinite(self.waveform).all():
            raise ValueError("waveform contains NaN or Inf")
