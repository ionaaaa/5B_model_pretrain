"""Dataset-independent standard-64 FBMAE runtime processing."""

from __future__ import annotations

import math
import time
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import spectrogram

from channel_config import CHANNEL_ALIASES, STANDARD_64_CHANNELS
from configs import Config
from runtime_adaptation.channel_mapping import resolve_channel_mapping_policy
from runtime_adaptation.contracts import CanonicalEEGWindow


def _next_power_of_two(value: int) -> int:
    return 1 << int(math.ceil(math.log2(value)))


def canonicalize_channel_name(name: str) -> str:
    stripped = str(name).strip()
    alias = CHANNEL_ALIASES.get(stripped, CHANNEL_ALIASES.get(stripped.upper(), stripped))
    exact = {channel: channel for channel in STANDARD_64_CHANNELS}
    if alias in exact:
        return alias
    casefold_lookup = {channel.casefold(): channel for channel in STANDARD_64_CHANNELS}
    return casefold_lookup.get(alias.casefold(), alias)


def align_window_to_standard64(
    window: CanonicalEEGWindow,
    standard_channels: Sequence[str] = STANDARD_64_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], str]:
    """Map one canonical source window to fixed standard-channel slots."""
    policy = resolve_channel_mapping_policy(window)
    standard_index = {name: index for index, name in enumerate(standard_channels)}
    signal = np.zeros((len(standard_channels), window.waveform.shape[1]), dtype=np.float32)
    channel_valid = np.zeros(len(standard_channels), dtype=np.float32)
    occupied = set()
    dropped = []

    for source_index, source_name in enumerate(window.channel_names):
        name = canonicalize_channel_name(source_name)
        if name not in standard_index:
            if source_name in policy.allowed_dropped_channels:
                dropped.append(source_name)
                continue
            raise ValueError(
                f"{window.dataset_key} channel {source_name!r} canonicalized to "
                f"{name!r}, which is absent from STANDARD_64_CHANNELS and is not "
                f"allowed by mapping policy {policy.name!r}"
            )
        destination = standard_index[name]
        if destination in occupied:
            raise ValueError(
                f"Duplicate {window.dataset_key} mapping to standard channel {name!r}"
            )
        occupied.add(destination)
        if window.channel_valid[source_index]:
            signal[destination] = window.waveform[source_index]
            channel_valid[destination] = 1.0

    return signal, channel_valid, tuple(dropped), policy.name


def compute_runtime_targets(
    signal64: np.ndarray,
    channel_valid_mask: np.ndarray,
    data_config=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the existing offline FBMAE time-frequency target pipeline."""
    cfg = Config().data if data_config is None else data_config
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


class SharedFBMAECore:
    """Convert a decoded canonical window into the common FBMAE row contract."""

    def __init__(self, data_config=None):
        self.cfg = Config().data if data_config is None else data_config

    def adapt(self, window: CanonicalEEGWindow, validate: bool = True) -> Dict[str, object]:
        expected_samples = int(round(self.cfg.clip_seconds * self.cfg.target_sfreq))
        if not np.isclose(window.sfreq, float(self.cfg.target_sfreq)):
            raise ValueError(
                f"SharedFBMAECore currently requires decoded sfreq="
                f"{self.cfg.target_sfreq}, got {window.sfreq} for {window.dataset_key}. "
                "Resampling policy must be established in the dataset adapter first."
            )
        if window.valid_samples != expected_samples:
            raise ValueError(
                f"Only full {expected_samples}-sample windows are supported, got "
                f"valid_samples={window.valid_samples} for {window.dataset_key}"
            )
        if window.waveform.shape[1] != expected_samples:
            raise ValueError(
                f"Expected waveform [C,{expected_samples}], got {window.waveform.shape}"
            )

        adaptation_started = time.perf_counter()
        (
            signal64,
            channel_valid_mask,
            dropped_channel_names,
            channel_mapping_policy,
        ) = align_window_to_standard64(window)
        patch_len = int(round(self.cfg.patch_seconds * self.cfg.target_sfreq))
        patch_stride = int(round(self.cfg.patch_stride_seconds * self.cfg.target_sfreq))
        n_patches = 1 + (signal64.shape[1] - patch_len) // patch_stride
        patch4d = np.stack(
            [
                signal64[:, index * patch_stride:index * patch_stride + patch_len]
                for index in range(n_patches)
            ],
            axis=1,
        ).astype(np.float32)
        token_inputs = patch4d.reshape(-1, patch_len)
        n_channels = len(STANDARD_64_CHANNELS)
        token_channel_indices = np.repeat(np.arange(n_channels), n_patches).astype(np.int64)
        token_time_indices = np.tile(np.arange(n_patches), n_channels).astype(np.int64)
        token_valid_mask = channel_valid_mask[token_channel_indices].astype(np.float32)
        adaptation_ms = (time.perf_counter() - adaptation_started) * 1000.0

        target_started = time.perf_counter()
        targets, target4d = compute_runtime_targets(
            signal64,
            channel_valid_mask,
            data_config=self.cfg,
        )
        target_ms = (time.perf_counter() - target_started) * 1000.0

        if validate:
            self._validate(
                signal64=signal64,
                patch4d=patch4d,
                target4d=target4d,
                token_inputs=token_inputs,
                targets=targets,
                token_channel_indices=token_channel_indices,
                token_time_indices=token_time_indices,
                token_valid_mask=token_valid_mask,
                channel_valid_mask=channel_valid_mask,
                patch_len=patch_len,
                n_patches=n_patches,
            )

        valid_waveform = signal64[channel_valid_mask.astype(bool)]
        if valid_waveform.size:
            waveform_mean_abs_max = float(np.max(np.abs(valid_waveform.mean(axis=1))))
            waveform_std_error_max = float(np.max(np.abs(valid_waveform.std(axis=1) - 1.0)))
        else:
            waveform_mean_abs_max = float("nan")
            waveform_std_error_max = float("nan")

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
            "source_channel_count": len(window.channel_names),
            "dropped_channel_count": len(dropped_channel_names),
            "dropped_channel_names": ",".join(dropped_channel_names),
            "channel_mapping_policy": channel_mapping_policy,
            "waveform_mean_abs_max": waveform_mean_abs_max,
            "waveform_std_error_max": waveform_std_error_max,
            "source_index": window.source_index,
            "sample_id": window.sample_id,
            "subject": window.subject_id,
            "dataset_id": window.dataset_key,
        }

    @staticmethod
    def _validate(
        *,
        signal64,
        patch4d,
        target4d,
        token_inputs,
        targets,
        token_channel_indices,
        token_time_indices,
        token_valid_mask,
        channel_valid_mask,
        patch_len,
        n_patches,
    ):
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

        expected_tokens = len(STANDARD_64_CHANNELS) * n_patches
        if token_inputs.shape != (expected_tokens, patch_len):
            raise AssertionError(f"Unexpected token_inputs shape: {token_inputs.shape}")
        if targets.shape[0] != expected_tokens:
            raise AssertionError(f"Unexpected targets shape: {targets.shape}")
        expected_channels = np.repeat(np.arange(len(STANDARD_64_CHANNELS)), n_patches)
        expected_times = np.tile(np.arange(n_patches), len(STANDARD_64_CHANNELS))
        if not np.array_equal(token_channel_indices, expected_channels):
            raise AssertionError("token ordering is not channel-major")
        if not np.array_equal(token_time_indices, expected_times):
            raise AssertionError("token time ordering is not channel-major")
        if not np.array_equal(token_inputs.reshape(patch4d.shape), patch4d):
            raise AssertionError("token_inputs flatten order does not match waveform patches")
        if not np.array_equal(targets.reshape(target4d.shape), target4d):
            raise AssertionError("target flatten order does not match token_inputs order")

        invalid = channel_valid_mask == 0
        if np.any(signal64[invalid] != 0) or np.any(patch4d[invalid] != 0):
            raise AssertionError("invalid channel waveform/token data is not zero")
        if np.any(target4d[invalid] != 0):
            raise AssertionError("invalid channel targets are not zero")
        valid_targets = target4d[channel_valid_mask.astype(bool)]
        if valid_targets.size == 0 or float(valid_targets.std()) <= 1e-6:
            raise AssertionError("targets are constant or no valid targets exist")
