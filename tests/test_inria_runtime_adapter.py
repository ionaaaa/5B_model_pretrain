import json
import unittest

import numpy as np

from inria_runtime_dataset import INRIA_CHANNELS, adapt_inria_row


def make_row(invalid_channel=None):
    sfreq = 100
    times = np.arange(1000, dtype=np.float32) / sfreq
    channels = []
    for index in range(len(INRIA_CHANNELS)):
        signal = (
            np.sin(2 * np.pi * (2.0 + index * 0.25) * times)
            + 0.25 * np.sin(2 * np.pi * (8.0 + index * 0.1) * times)
        ).astype(np.float32)
        signal = (signal - signal.mean()) / signal.std()
        channels.append(signal)
    eeg = np.stack(channels)
    mask = np.ones(len(INRIA_CHANNELS), dtype=np.int64)
    if invalid_channel is not None:
        mask[invalid_channel] = 0
    return {
        "eeg": eeg.reshape(-1).tolist(),
        "eeg_shape": [len(INRIA_CHANNELS), 1000],
        "eeg_n_channels": len(INRIA_CHANNELS),
        "eeg_channel_indices": list(range(len(INRIA_CHANNELS))),
        "valid_channel_mask_json": json.dumps(mask.tolist()),
        "valid_samples": 1000,
        "source_index": 7,
        "subject": "synthetic",
        "dataset_id": "inria",
    }


class TestInriaRuntimeAdapter(unittest.TestCase):
    def test_shapes_ordering_and_targets(self):
        sample = adapt_inria_row(make_row())
        self.assertEqual(tuple(sample["token_inputs"].shape), (640, 100))
        self.assertEqual(tuple(sample["targets"].shape), (640, 5, 5))
        self.assertEqual(tuple(sample["token_channel_indices"].shape), (640,))
        self.assertEqual(tuple(sample["token_time_indices"].shape), (640,))
        self.assertEqual(tuple(sample["token_valid_mask"].shape), (640,))
        self.assertEqual(tuple(sample["channel_valid_mask"].shape), (64,))
        np.testing.assert_array_equal(
            sample["token_channel_indices"].numpy()[:20],
            np.repeat(np.arange(2), 10),
        )
        np.testing.assert_array_equal(
            sample["token_time_indices"].numpy()[:20],
            np.tile(np.arange(10), 2),
        )
        self.assertTrue(np.isfinite(sample["token_inputs"].numpy()).all())
        self.assertTrue(np.isfinite(sample["targets"].numpy()).all())
        self.assertGreater(float(sample["targets"].std()), 1e-6)
        self.assertEqual(sample["valid_channel_count"], len(INRIA_CHANNELS))

    def test_invalid_channel_is_zero_in_tokens_targets_and_masks(self):
        invalid_source_channel = 0
        sample = adapt_inria_row(make_row(invalid_channel=invalid_source_channel))
        fz_slot = 12
        self.assertEqual(float(sample["channel_valid_mask"][fz_slot]), 0.0)
        self.assertTrue(np.all(sample["token_inputs"][fz_slot * 10:(fz_slot + 1) * 10].numpy() == 0))
        self.assertTrue(np.all(sample["targets"][fz_slot * 10:(fz_slot + 1) * 10].numpy() == 0))
        self.assertTrue(np.all(sample["token_valid_mask"][fz_slot * 10:(fz_slot + 1) * 10].numpy() == 0))

    def test_rejects_partial_window(self):
        row = make_row()
        row["valid_samples"] = 999
        with self.assertRaisesRegex(ValueError, "valid_samples == 1000"):
            adapt_inria_row(row)


if __name__ == "__main__":
    unittest.main()
