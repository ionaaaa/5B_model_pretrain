import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from configs import Config, apply_pretrain_profile
from fsdp_utils import (
    fsdp_auto_wrap_policy,
    fsdp_mixed_precision_policy,
    fresh_training_state,
    load_model_state_dict,
    restore_training_checkpoint,
    save_model_checkpoint,
)
from losses import BalancedBandTokenTrajectoryLoss
from model import EEGPretrainModel


TINY_MODEL_KWARGS = {
    "input_dim": 8,
    "d_model": 64,
    "n_heads": 4,
    "depth": 2,
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "n_channels": 4,
    "n_time_patches": 3,
    "n_bands": 5,
    "frames_per_patch": 4,
}


def build_tiny_model():
    return EEGPretrainModel(**TINY_MODEL_KWARGS)


def make_tiny_batch(batch_size=2):
    num_tokens = TINY_MODEL_KWARGS["n_channels"] * TINY_MODEL_KWARGS["n_time_patches"]
    channel_indices = torch.arange(4).repeat_interleave(3).expand(batch_size, -1)
    time_indices = torch.arange(3).repeat(4).expand(batch_size, -1)
    return {
        "token_inputs": torch.randn(batch_size, num_tokens, 8),
        "targets": torch.randn(batch_size, num_tokens, 5, 4),
        "token_channel_indices": channel_indices,
        "token_time_indices": time_indices,
        "token_valid_mask": torch.ones(batch_size, num_tokens),
    }


def run_tiny_step(model, optimizer, batch):
    pred = model(
        token_inputs=batch["token_inputs"],
        token_channel_indices=batch["token_channel_indices"],
        token_time_indices=batch["token_time_indices"],
    )
    loss = BalancedBandTokenTrajectoryLoss()(
        pred,
        batch["targets"],
        batch["token_valid_mask"],
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.item()


class TestFSDPSmoke(unittest.TestCase):
    def test_formal_default_is_50m_without_constructing_a_large_model(self):
        cfg = Config()
        self.assertEqual((cfg.model.d_model, cfg.model.n_heads), (512, 8))
        self.assertEqual((cfg.model.depth, cfg.model.mlp_ratio), (12, 4.0))
        self.assertEqual(cfg.model.d_model // cfg.model.n_heads, 64)

    def test_five_billion_profile_remains_selectable_without_constructing_it(self):
        cfg = apply_pretrain_profile(Config(), "5b")
        self.assertEqual((cfg.model.d_model, cfg.model.n_heads), (4096, 16))
        self.assertEqual((cfg.model.depth, cfg.model.mlp_ratio), (25, 4.0))
        self.assertEqual(cfg.model.d_model // cfg.model.n_heads, 256)

    def test_tiny_real_eeg_model_forward_backward_and_optimizer_step(self):
        torch.manual_seed(0)
        model = build_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss = run_tiny_step(model, optimizer, make_tiny_batch())
        self.assertGreater(loss, 0.0)

    def test_auto_wrap_policy_targets_the_real_transformer_block(self):
        policy = fsdp_auto_wrap_policy()
        self.assertEqual(
            policy.keywords["transformer_layer_cls"],
            {nn.TransformerEncoderLayer},
        )

    def test_production_mixed_precision_policy_is_bfloat16(self):
        policy = fsdp_mixed_precision_policy()
        self.assertEqual(policy.param_dtype, torch.bfloat16)
        self.assertEqual(policy.reduce_dtype, torch.bfloat16)
        self.assertEqual(policy.buffer_dtype, torch.bfloat16)

    def test_training_entry_wraps_before_constructing_adamw(self):
        source = Path("largeScale_pretrain_main.py").read_text(encoding="utf-8")
        self.assertNotIn("DistributedDataParallel", source)
        self.assertLess(
            source.index("model = build_fsdp_model(raw_model, device=device)"),
            source.index("optimizer = torch.optim.AdamW("),
        )

    def test_training_entry_saves_initial_and_per_epoch_artifacts(self):
        source = Path("largeScale_pretrain_main.py").read_text(encoding="utf-8")
        self.assertIn('f"{base_name}_initial.pt"', source)
        self.assertIn('artifact_tag="before_training"', source)
        self.assertIn('artifact_tag=f"epoch_{epoch + 1:03d}"', source)
        self.assertIn('artifact_tag="final"', source)

    def test_training_entry_exposes_resume_cli(self):
        source = Path("largeScale_pretrain_main.py").read_text(encoding="utf-8")
        self.assertIn('"--resume"', source)
        self.assertIn("range(start_epoch, cfg.train.num_epochs)", source)

        from largeScale_pretrain_main import parse_args

        with patch("sys.argv", ["largeScale_pretrain_main.py", "--resume", "epoch0.pt"]):
            self.assertEqual(parse_args().resume, "epoch0.pt")

    def test_model_only_checkpoint_restores_model_and_training_progress(self):
        torch.manual_seed(1)
        model = build_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        run_tiny_step(model, optimizer, make_tiny_batch())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "tiny.pt")
            save_model_checkpoint(
                model=model,
                path=checkpoint_path,
                rank=0,
                extra_state={
                    "epoch": 1,
                    "global_step": 28740,
                    "best_val": 0.5,
                    "train_loss": 0.4,
                    "val_loss": 0.3,
                },
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            restored_model = build_tiny_model()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_state = restore_training_checkpoint(
                model=restored_model,
                optimizer=restored_optimizer,
                path=checkpoint_path,
                steps_per_epoch=5,
            )
        self.assertEqual(checkpoint["checkpoint_format"], "model_only")
        self.assertEqual(checkpoint["checkpoint_version"], 3)
        self.assertNotIn("optimizer_state_dict", checkpoint)
        self.assertNotIn("scheduler_state_dict", checkpoint)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["global_step"], 28740)
        self.assertFalse(restored_state["optimizer_restored"])
        self.assertEqual(restored_state["start_epoch"], 2)
        self.assertEqual(restored_state["global_step"], 28740)
        self.assertTrue(any("optimizer state" in warning for warning in restored_state["warnings"]))
        for expected, restored in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(expected, restored)

        self.assertGreater(
            run_tiny_step(restored_model, restored_optimizer, make_tiny_batch()),
            0.0,
        )

    def test_legacy_model_only_checkpoint_without_global_step_is_compatible(self):
        model = build_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        run_tiny_step(model, optimizer, make_tiny_batch())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "legacy.pt")
            torch.save({"model_state_dict": model.state_dict(), "epoch": 0}, checkpoint_path)
            restored_model = build_tiny_model()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_state = restore_training_checkpoint(
                model=restored_model,
                optimizer=restored_optimizer,
                path=checkpoint_path,
                steps_per_epoch=7,
            )

        self.assertEqual(restored_state["checkpoint_version"], 1)
        self.assertEqual(restored_state["start_epoch"], 1)
        self.assertEqual(restored_state["global_step"], 7)
        self.assertFalse(restored_state["optimizer_restored"])
        self.assertTrue(any("optimizer state" in warning for warning in restored_state["warnings"]))
        self.assertTrue(any("global_step" in warning for warning in restored_state["warnings"]))
        for expected, restored in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(expected, restored)

    def test_legacy_full_checkpoint_restores_optimizer_state(self):
        model = build_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        run_tiny_step(model, optimizer, make_tiny_batch())

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "legacy_full.pt")
            torch.save(
                {
                    "checkpoint_version": 2,
                    "checkpoint_format": "fsdp_full_model_state_dict_rank0",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": 1,
                    "global_step": 28740,
                },
                checkpoint_path,
            )
            restored_model = build_tiny_model()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_state = restore_training_checkpoint(
                model=restored_model,
                optimizer=restored_optimizer,
                path=checkpoint_path,
                steps_per_epoch=5,
            )

        self.assertTrue(restored_state["optimizer_restored"])
        self.assertEqual(restored_state["start_epoch"], 2)
        self.assertEqual(restored_state["global_step"], 28740)
        self.assertGreater(len(restored_optimizer.state), 0)
        for expected, restored in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(expected, restored)

    def test_fresh_training_state_starts_at_epoch_zero(self):
        state = fresh_training_state()
        self.assertEqual(state["start_epoch"], 0)
        self.assertEqual(state["global_step"], 0)

    def test_cpu_fsdp_runtime_is_not_supported_by_this_torch_version(self):
        # PyTorch 2.0 FSDP's compute-device implementation explicitly selects CUDA
        # for CPU-resident modules, so this environment cannot run an FSDP runtime
        # smoke test without a GPU. Production code remains CUDA/FSDP-only.
        import torch.distributed.fsdp._init_utils as init_utils

        source = inspect.getsource(init_utils._get_compute_device)
        self.assertIn('torch.device("cuda", torch.cuda.current_device())', source)


if __name__ == "__main__":
    unittest.main()
