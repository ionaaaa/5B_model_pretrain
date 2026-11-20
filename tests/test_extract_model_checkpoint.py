import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from configs import Config


SCRIPT = Path("scripts/extract_model_from_checkpoint.py").resolve()


class ExtractModelCheckpointTests(unittest.TestCase):
    def test_extracts_model_and_excludes_optimizer_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "full.pt"
            output_path = Path(temp_dir) / "model_only.pt"
            model_state_dict = {
                "linear.weight": torch.randn(4, 4),
                "linear.bias": torch.randn(4),
            }
            torch.save(
                {
                    "checkpoint_version": 2,
                    "checkpoint_format": "full_model_state_dict",
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": {"state": {"dummy": "dummy"}},
                    "scheduler_state_dict": {"last_epoch": 1},
                    "config": Config(),
                    "epoch": 1,
                    "global_step": 100,
                    "best_val": 0.25,
                },
                input_path,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--verify",
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=temp_dir,
            )
            converted = torch.load(output_path, map_location="cpu", weights_only=False)

        self.assertIn("Model-only checkpoint successfully saved", result.stdout)
        self.assertTrue(output_path.name.endswith(".pt"))
        self.assertEqual(converted["checkpoint_format"], "model_only")
        self.assertEqual(converted["epoch"], 1)
        self.assertEqual(converted["global_step"], 100)
        self.assertEqual(tuple(converted["model_state_dict"].keys()), tuple(model_state_dict.keys()))
        self.assertNotIn("optimizer_state_dict", converted)
        self.assertNotIn("scheduler_state_dict", converted)

    def test_refuses_to_overwrite_without_flag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "full.pt"
            output_path = Path(temp_dir) / "model_only.pt"
            torch.save({"model_state_dict": {"weight": torch.ones(1)}}, input_path)
            torch.save({"existing": True}, output_path)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Output checkpoint already exists", result.stderr)


if __name__ == "__main__":
    unittest.main()
