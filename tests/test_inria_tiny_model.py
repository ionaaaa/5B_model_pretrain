import json
import types
import unittest

import numpy as np
import torch

from inria_runtime_dataset import INRIA_CHANNELS, adapt_inria_row
from scripts.validate_inria_tiny_model import run_tiny_step


def _synthetic_runtime_batch():
    rng = np.random.default_rng(20261121)
    row = {
        "eeg": rng.standard_normal((len(INRIA_CHANNELS), 1000)).astype(np.float32).reshape(-1),
        "eeg_shape": [len(INRIA_CHANNELS), 1000],
        "eeg_n_channels": len(INRIA_CHANNELS),
        "eeg_channel_indices": list(range(len(INRIA_CHANNELS))),
        "valid_channel_mask_json": json.dumps([1] * len(INRIA_CHANNELS)),
        "valid_samples": "1000",
        "source_index": "0",
        "subject": "synthetic",
        "dataset_id": "inria",
    }
    sample = adapt_inria_row(row, validate=True)
    return {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if torch.is_tensor(value)
    }


class TestInriaTinyFormalModel(unittest.TestCase):
    def test_runtime_adapter_to_formal_model_optimizer_step(self):
        args = types.SimpleNamespace(
            device="cpu",
            seed=20261121,
            d_model=64,
            n_heads=4,
            depth=2,
            mask_ratio=0.8,
        )
        metrics = run_tiny_step(_synthetic_runtime_batch(), args)
        self.assertEqual(metrics["pred_shape"], (1, 640, 5, 5))
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(np.isfinite(metrics["gradient_norm"]))
        self.assertGreater(metrics["gradient_tensors"], 0)
        self.assertGreater(metrics["changed_parameter_tensors"], 0)
        self.assertEqual(metrics["masked_tokens_per_sample"], [512.0])


if __name__ == "__main__":
    unittest.main()
