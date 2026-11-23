import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from configs import Config, MaskConfig, apply_pretrain_profile
from losses import BalancedBandTokenTrajectoryLoss
from masking import expected_mask_ratio, sample_mask_ratio, validate_mask_config
import trainer


class TestModelScale(unittest.TestCase):
    def test_default_config_is_the_50m_trial_shape(self):
        cfg = Config()
        self.assertEqual(cfg.model.profile, "50m")
        self.assertEqual(cfg.model.d_model, 512)
        self.assertEqual(cfg.model.n_heads, 8)
        self.assertEqual(cfg.model.depth, 12)
        self.assertEqual(cfg.model.mlp_ratio, 4.0)
        self.assertEqual(cfg.model.d_model // cfg.model.n_heads, 64)

    def test_five_billion_profile_is_retained(self):
        cfg = apply_pretrain_profile(Config(), "5b")
        self.assertEqual(cfg.model.profile, "5b")
        self.assertEqual(cfg.model.d_model, 4096)
        self.assertEqual(cfg.model.n_heads, 16)
        self.assertEqual(cfg.model.depth, 25)
        self.assertEqual(cfg.train.batch_size, 32)
        self.assertEqual(cfg.train.lr, 5e-5)


class TestDynamicMasking(unittest.TestCase):
    def test_sampler_returns_only_configured_values(self):
        cfg = MaskConfig()
        torch.manual_seed(7)
        samples = [sample_mask_ratio(cfg) for _ in range(1000)]
        self.assertEqual(set(samples), {0.9})

    def test_default_expected_ratio_is_fixed_at_point_nine(self):
        cfg = MaskConfig()
        self.assertAlmostEqual(expected_mask_ratio(cfg), 0.9)

    def test_invalid_mask_configs_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_mask_config(MaskConfig(mask_ratios=[0.6], mask_ratio_weights=[]))
        with self.assertRaisesRegex(ValueError, "0 <= mask_ratio < 1"):
            validate_mask_config(MaskConfig(mask_ratios=[1.0], mask_ratio_weights=[1.0]))
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            validate_mask_config(MaskConfig(mask_ratios=[0.6], mask_ratio_weights=[0.0]))

    def test_training_resamples_once_per_batch_not_per_sample(self):
        cfg = Config()
        cfg.train.print_every = 100
        num_batches = 5
        batch_size = 32
        loader = [_make_batch(batch_size=batch_size) for _ in range(num_batches)]
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        sampled_ratios = [0.6, 0.7, 0.8, 0.9, 0.6]
        with patch("trainer.sample_mask_ratio", side_effect=sampled_ratios) as sampler:
            with patch(
                "trainer.build_batch_token_mask",
                wraps=trainer.build_batch_token_mask,
            ) as build_mask:
                _, global_step = trainer.train_one_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    criterion=_TinyCriterion(),
                    device=torch.device("cpu"),
                    cfg=cfg,
                )

        self.assertEqual(sampler.call_count, num_batches)
        self.assertEqual(build_mask.call_count, num_batches)
        self.assertEqual(global_step, num_batches)
        self.assertEqual(
            [call.kwargs["mask_ratio"] for call in build_mask.call_args_list],
            sampled_ratios,
        )

    def test_reconstruction_loss_remains_over_all_valid_tokens(self):
        criterion = BalancedBandTokenTrajectoryLoss()
        pred = torch.tensor([[[[0.0]], [[0.0]]]])
        target = torch.tensor([[[[1.0]], [[3.0]]]])
        token_valid_mask = torch.tensor([[1.0, 1.0]])

        # No random token mask is passed to the criterion: both valid tokens count.
        self.assertEqual(criterion(pred, target, token_valid_mask).item(), 5.0)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, token_inputs, token_channel_indices, token_time_indices):
        return self.scale * token_inputs[:, :, :1].unsqueeze(-1)


class _TinyCriterion(nn.Module):
    def forward(self, pred, target, token_valid_mask):
        valid = token_valid_mask[:, :, None, None]
        return (((pred - target) ** 2) * valid).mean()


def _make_batch(batch_size):
    num_tokens = 10
    return {
        "token_inputs": torch.ones(batch_size, num_tokens, 1),
        "targets": torch.zeros(batch_size, num_tokens, 1, 1),
        "token_channel_indices": torch.zeros(batch_size, num_tokens, dtype=torch.long),
        "token_time_indices": torch.zeros(batch_size, num_tokens, dtype=torch.long),
        "token_valid_mask": torch.ones(batch_size, num_tokens),
    }


if __name__ == "__main__":
    unittest.main()
