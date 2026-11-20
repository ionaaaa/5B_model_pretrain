import sys
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from largeScale_pretrain_main import (
    maybe_wrap_prefetch_loaders,
    parse_args,
    resolve_use_prefetch_loader,
)
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


if __name__ == "__main__":
    unittest.main()
