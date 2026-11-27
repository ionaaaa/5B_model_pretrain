import json
import os
from unittest.mock import patch

from configs import Config
from largeScale_pretrain_main import (
    _preflight_coordination_path,
    prepare_processed_preflight_before_distributed,
)


def _distributed_env(rank):
    return {
        "RANK": str(rank),
        "WORLD_SIZE": "8",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29671",
        "TORCHELASTIC_RUN_ID": "preflight-unit-test",
    }


def _report():
    return {
        "parts": 2,
        "total_rows": 3,
        "sampled_rows": 2,
        "lengths": [2, 1],
        "expected_shapes": {},
        "part_reports": [],
        "elapsed_seconds": 0.1,
    }


def test_rank_zero_runs_preflight_and_atomically_publishes_report(tmp_path):
    cfg = Config()
    cfg.data.lance_part_list_file = str(tmp_path / "parts.txt")

    with patch.dict(os.environ, _distributed_env(rank=0), clear=False):
        with patch(
            "largeScale_pretrain_main._run_processed_lance_preflight",
            return_value=_report(),
        ) as run_preflight:
            result = prepare_processed_preflight_before_distributed(cfg)
        report_path = _preflight_coordination_path(cfg)

    assert result == _report()
    run_preflight.assert_called_once_with(cfg)
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as handle:
        assert json.load(handle) == {"report": _report()}


def test_nonzero_rank_reads_shared_report_without_running_lance_scan(tmp_path):
    cfg = Config()
    cfg.data.lance_part_list_file = str(tmp_path / "parts.txt")

    with patch.dict(os.environ, _distributed_env(rank=1), clear=False):
        report_path = _preflight_coordination_path(cfg)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump({"report": _report()}, handle)

        with patch(
            "largeScale_pretrain_main._run_processed_lance_preflight",
            side_effect=AssertionError("nonzero ranks must not scan Lance"),
        ) as run_preflight:
            result = prepare_processed_preflight_before_distributed(cfg)

    assert result == _report()
    run_preflight.assert_not_called()
