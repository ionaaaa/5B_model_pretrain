import io
import os
import re
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch
import torch.nn as nn

from configs import Config
from largeScale_pretrain_main import (
    NCCL_FLIGHT_RECORDER_DEFAULTS,
    configure_nccl_flight_recorder_defaults,
    log_nccl_flight_recorder_status,
)
import trainer


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


def _make_batch():
    num_tokens = 10
    return {
        "token_inputs": torch.ones(1, num_tokens, 1),
        "targets": torch.zeros(1, num_tokens, 1, 1),
        "token_channel_indices": torch.zeros(1, num_tokens, dtype=torch.long),
        "token_time_indices": torch.zeros(1, num_tokens, dtype=torch.long),
        "token_valid_mask": torch.ones(1, num_tokens),
    }


class NCCLDiagnosticTests(unittest.TestCase):
    def test_flight_recorder_defaults_preserve_explicit_environment(self):
        with patch.dict(
            os.environ,
            {"TORCH_NCCL_TRACE_BUFFER_SIZE": "5000"},
            clear=True,
        ):
            configure_nccl_flight_recorder_defaults()
            self.assertEqual(os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"], "5000")
            for name, value in NCCL_FLIGHT_RECORDER_DEFAULTS.items():
                self.assertIn(name, os.environ)
                if name != "TORCH_NCCL_TRACE_BUFFER_SIZE":
                    self.assertEqual(os.environ[name], value)

    def test_flight_recorder_status_reports_effective_values(self):
        with patch.dict(os.environ, {}, clear=True):
            output = io.StringIO()
            with redirect_stdout(output):
                log_nccl_flight_recorder_status(rank=0)

        text = output.getvalue()
        self.assertIn("[NCCL Debug]", text)
        for name in NCCL_FLIGHT_RECORDER_DEFAULTS:
            self.assertIn(f"{name}=<unset>", text)

    def test_step_debug_is_disabled_at_zero(self):
        cfg = Config()
        cfg.train.print_every = 1000
        cfg.train.step_debug_last_n = 0
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        output = io.StringIO()
        with redirect_stdout(output):
            trainer.train_one_epoch(
                model=model,
                loader=[_make_batch() for _ in range(3)],
                optimizer=optimizer,
                criterion=_TinyCriterion(),
                device=torch.device("cpu"),
                cfg=cfg,
                epoch=0,
            )

        self.assertNotIn("[StepDebug]", output.getvalue())

    def test_step_debug_marks_only_final_batches_and_all_three_states(self):
        cfg = Config()
        cfg.train.print_every = 1000
        cfg.train.step_debug_last_n = 5
        model = _TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        output = io.StringIO()
        with redirect_stdout(output):
            trainer.train_one_epoch(
                model=model,
                loader=[_make_batch() for _ in range(100)],
                optimizer=optimizer,
                criterion=_TinyCriterion(),
                device=torch.device("cpu"),
                cfg=cfg,
                global_step=1000,
                epoch=3,
            )

        markers = [line for line in output.getvalue().splitlines() if "[StepDebug]" in line]
        self.assertEqual(len(markers), 15)
        for state in ("batch_ready", "before_forward", "after_forward"):
            self.assertEqual(sum(f"state={state}" in line for line in markers), 5)
        displayed_steps = {int(match.group(1)) for line in markers if (match := re.search(r"step=(\d+)", line))}
        self.assertEqual(displayed_steps, {96, 97, 98, 99, 100})
        self.assertTrue(all("epoch=4" in line for line in markers))

    def test_step_debug_logger_has_no_collective_or_cuda_sync(self):
        source = trainer.log_step_debug.__code__.co_names
        self.assertNotIn("barrier", source)
        self.assertNotIn("all_reduce", source)
        self.assertNotIn("synchronize", source)


if __name__ == "__main__":
    unittest.main()
