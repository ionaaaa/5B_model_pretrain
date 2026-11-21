import json
import types
import unittest

import numpy as np
import torch

from inria_runtime_dataset import INRIA_CHANNELS, adapt_inria_row
from scripts.validate_runtime_short_overfit import (
    _reject_unready_dataset,
    _validate_cached_samples,
    run_short_overfit,
)


def _make_cached_samples(sample_count=4):
    samples = []
    time = np.arange(1000, dtype=np.float32) / 100.0
    for sample_index in range(sample_count):
        channels = []
        for channel_index in range(len(INRIA_CHANNELS)):
            frequency = 2.0 + ((sample_index + channel_index) % 20)
            signal = np.sin(2.0 * np.pi * frequency * time)
            signal += 0.15 * np.cos(2.0 * np.pi * (frequency + 3.0) * time)
            channels.append(signal.astype(np.float32))
        waveform = np.stack(channels)
        row = {
            "eeg": waveform.reshape(-1),
            "eeg_shape": [len(INRIA_CHANNELS), 1000],
            "eeg_n_channels": len(INRIA_CHANNELS),
            "eeg_channel_indices": list(range(len(INRIA_CHANNELS))),
            "valid_channel_mask_json": json.dumps([1] * len(INRIA_CHANNELS)),
            "valid_samples": "1000",
            "source_index": str(sample_index),
            "subject": f"synthetic-{sample_index}",
            "dataset_id": "inria",
        }
        samples.append(adapt_inria_row(row, validate=True))
    fields = (
        "token_inputs",
        "targets",
        "token_channel_indices",
        "token_time_indices",
        "token_valid_mask",
        "channel_valid_mask",
    )
    return {
        field: torch.stack([sample[field] for sample in samples])
        for field in fields
    }


class TestRuntimeShortOverfit(unittest.TestCase):
    def test_dataset_gates(self):
        _reject_unready_dataset("inria")
        with self.assertRaisesRegex(RuntimeError, "no validated runtime"):
            _reject_unready_dataset("egobrain")
        with self.assertRaisesRegex(RuntimeError, "electrode coordinate"):
            _reject_unready_dataset("ds004408")

    def test_short_overfit_engine_stays_finite(self):
        data = _make_cached_samples()
        stats = _validate_cached_samples(data)
        self.assertEqual(stats["valid_tokens_min"], 270)
        self.assertGreater(stats["target_std"], 0.0)

        args = types.SimpleNamespace(
            device="cpu",
            seed=20261121,
            d_model=32,
            n_heads=4,
            depth=1,
            mask_ratio=0.5,
            learning_rate=1e-3,
            eval_every=1,
            steps=2,
            batch_size=2,
            min_loss_drop=-1.0,
            min_zero_baseline_advantage=-1.0,
            min_shuffle_advantage=-1.0,
        )
        result = run_short_overfit(data, args)
        self.assertTrue(np.isfinite(result["aligned"]["final_loss"]))
        self.assertTrue(np.isfinite(result["shuffled"]["final_loss"]))


if __name__ == "__main__":
    unittest.main()
