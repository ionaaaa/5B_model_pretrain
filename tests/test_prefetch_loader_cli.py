import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from largeScale_pretrain_main import (
    apply_cli_config,
    maybe_wrap_prefetch_loaders,
    parse_args,
    resolve_use_prefetch_loader,
)
from configs import Config
from prefetch_loader import PrefetchLoader


class PrefetchLoaderCLITests(unittest.TestCase):
    def setUp(self):
        dataset = TensorDataset(torch.arange(8))
        self.train_loader = DataLoader(dataset, batch_size=2, num_workers=0)
        self.val_loader = DataLoader(dataset, batch_size=2, num_workers=0)

    def test_enabled_wraps_both_loaders(self):
        train_loader, val_loader = maybe_wrap_prefetch_loaders(
            self.train_loader,
            self.val_loader,
            use_prefetch_loader=True,
            device="cpu",
            prefetch=2,
        )
        self.assertIsInstance(train_loader, PrefetchLoader)
        self.assertIsInstance(val_loader, PrefetchLoader)
        self.assertIs(train_loader.loader, self.train_loader)
        self.assertIs(val_loader.loader, self.val_loader)

    def test_disabled_returns_the_original_raw_dataloaders(self):
        train_loader, val_loader = maybe_wrap_prefetch_loaders(
            self.train_loader,
            self.val_loader,
            use_prefetch_loader=False,
            device="cpu",
            prefetch=2,
        )
        self.assertIs(train_loader, self.train_loader)
        self.assertIs(val_loader, self.val_loader)
        self.assertIsInstance(train_loader, DataLoader)
        self.assertEqual(train_loader.num_workers, 0)
        self.assertEqual(train_loader.batch_size, 2)

    def test_cli_omission_keeps_the_config_value(self):
        with patch.object(sys, "argv", ["largeScale_pretrain_main.py"]):
            self.assertIsNone(parse_args().use_prefetch_loader)
        self.assertTrue(resolve_use_prefetch_loader(None, True))
        self.assertFalse(resolve_use_prefetch_loader(None, False))

    def test_explicit_cli_value_overrides_config_value(self):
        with patch.object(sys, "argv", ["largeScale_pretrain_main.py", "--no-prefetch-loader"]):
            self.assertFalse(parse_args().use_prefetch_loader)
        with patch.object(sys, "argv", ["largeScale_pretrain_main.py", "--use-prefetch-loader"]):
            self.assertTrue(parse_args().use_prefetch_loader)
        self.assertFalse(resolve_use_prefetch_loader(False, True))
        self.assertTrue(resolve_use_prefetch_loader(True, False))

    def test_formal_runtime_cli_applies_profile_then_run_overrides(self):
        argv = [
            "largeScale_pretrain_main.py",
            "--model-profile", "5b",
            "--data-mode", "canonical_runtime",
            "--dataset-key", "egobrain",
            "--lance-path", "/opt/ml/omni/datasets/egobrain/data.lance",
            "--max-samples", "128",
            "--batch-size", "4",
            "--num-workers", "2",
            "--num-epochs", "1",
            "--learning-rate", "0.0002",
            "--output-root", "/mnt/cfs/output/runtime_trial",
        ]
        with patch.object(sys, "argv", argv):
            cfg = apply_cli_config(Config(), parse_args())

        self.assertEqual(cfg.model.profile, "5b")
        self.assertEqual(cfg.model.d_model, 4096)
        self.assertEqual(cfg.model.n_heads, 16)
        self.assertEqual(cfg.model.depth, 25)
        self.assertEqual(cfg.mask.mask_ratios, [0.8])
        self.assertEqual(cfg.mask.mask_ratio_weights, [1.0])
        self.assertEqual(cfg.data.input_mode, "canonical_runtime")
        self.assertEqual(cfg.data.canonical_dataset_key, "egobrain")
        self.assertEqual(
            cfg.data.canonical_lance_path,
            "/opt/ml/omni/datasets/egobrain/data.lance",
        )
        self.assertEqual(cfg.data.runtime_max_samples, 128)
        self.assertEqual(cfg.train.batch_size, 4)
        self.assertEqual(cfg.train.num_workers, 2)
        self.assertEqual(cfg.train.num_epochs, 1)
        self.assertEqual(cfg.train.lr, 0.0002)
        self.assertEqual(cfg.train.output_root, "/mnt/cfs/output/runtime_trial")

    def test_processed_multilance_cli_sets_explicit_list_and_preflight_sample_size(self):
        part_list = (
            "/mnt/cfs/jianyu/5B_model_pretrain/"
            "processed_trial3_part_paths.txt"
        )
        argv = [
            "largeScale_pretrain_main.py",
            "--model-profile", "50m",
            "--data-mode", "processed_lance",
            "--part-list-file", part_list,
            "--preflight-rows-per-part", "4",
        ]
        with patch.object(sys, "argv", argv):
            cfg = apply_cli_config(Config(), parse_args())

        self.assertEqual(cfg.model.profile, "50m")
        self.assertEqual(cfg.data.input_mode, "processed_lance")
        self.assertTrue(cfg.data.use_multi_lance_parts)
        self.assertEqual(cfg.data.lance_part_list_file, part_list)
        self.assertEqual(cfg.data.processed_preflight_rows_per_part, 4)
        self.assertEqual(cfg.mask.mask_ratios, [0.8])


if __name__ == "__main__":
    unittest.main()
