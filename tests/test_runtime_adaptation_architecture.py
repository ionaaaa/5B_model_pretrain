import tempfile
import unittest
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch

from inria_runtime_dataset import (
    _legacy_adapt_inria_row_reference,
    adapt_inria_row,
)
from runtime_adaptation import CanonicalEEGWindow, InriaDecoder, SharedFBMAECore
from scripts.probe_canonical_lance_schemas import probe_dataset
from tests.test_inria_runtime_adapter import make_row


class TestRuntimeAdaptationArchitecture(unittest.TestCase):
    def test_inria_decoder_returns_canonical_contract(self):
        window = InriaDecoder().decode(make_row())
        self.assertIsInstance(window, CanonicalEEGWindow)
        self.assertEqual(window.waveform.shape, (27, 1000))
        self.assertEqual(len(window.channel_names), 27)
        self.assertEqual(window.channel_valid.shape, (27,))
        self.assertEqual(window.valid_samples, 1000)
        self.assertEqual(window.sfreq, 100.0)
        self.assertEqual(window.dataset_key, "inria")

    def test_new_decoder_core_path_exactly_matches_legacy_reference(self):
        row = make_row(invalid_channel=3)
        legacy = _legacy_adapt_inria_row_reference(row, validate=True)
        current = adapt_inria_row(row, validate=True)
        for field in (
            "token_inputs",
            "targets",
            "token_channel_indices",
            "token_time_indices",
            "token_valid_mask",
            "channel_valid_mask",
        ):
            self.assertTrue(torch.equal(current[field], legacy[field]), field)
        self.assertEqual(current["valid_channel_count"], legacy["valid_channel_count"])
        self.assertEqual(current["source_index"], legacy["source_index"])
        self.assertEqual(current["subject"], legacy["subject"])
        self.assertEqual(current["dataset_id"], legacy["dataset_id"])

    def test_shared_core_rejects_unestablished_resampling_policy(self):
        window = CanonicalEEGWindow(
            waveform=np.zeros((1, 1000), dtype=np.float32),
            channel_names=("Cz",),
            channel_valid=np.ones(1, dtype=bool),
            valid_samples=1000,
            sfreq=250.0,
            dataset_key="unverified",
        )
        with self.assertRaisesRegex(ValueError, "Resampling policy must be established"):
            SharedFBMAECore().adapt(window)

    def test_read_only_probe_recovers_physical_and_semantic_fingerprint(self):
        table = pa.table(
            {
                "eeg": [list(map(float, range(20))), list(map(float, range(20)))],
                "eeg_shape": [[2, 10], [2, 10]],
                "eeg_n_channels": [2, 2],
                "eeg_channel_indices": [[0, 1], [0, 1]],
                "valid_samples": ["10", "7"],
                "sfreq": [100.0, 100.0],
                "dataset_id": ["toy", "toy"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "toy.lance")
            lance.write_dataset(table, path, mode="overwrite")
            report = probe_dataset(
                "toy",
                path,
                head_rows=2,
                max_scan_rows=0,
                batch_size=16,
            )
        self.assertEqual(report["row_count"], 2)
        self.assertIn("list<", dict(
            (field["name"], field["physical_type"]) for field in report["fields"]
        )["eeg_shape"])
        self.assertEqual(report["semantics"]["shape_counts"], [((2, 10), 2)])
        self.assertEqual(
            sorted(report["semantics"]["valid_samples_counts"]),
            [(7, 1), (10, 1)],
        )
        self.assertTrue(report["sampling_rate_evidence"]["established_from_lance"])


if __name__ == "__main__":
    unittest.main()
