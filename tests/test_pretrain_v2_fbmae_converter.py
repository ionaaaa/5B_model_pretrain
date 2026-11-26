import json

import numpy as np
import lance
import pyarrow as pa

from build_fbmae_processed_from_pretrain_v2 import (
    N_CHANNELS,
    OUTPUT_SCHEMA,
    STANDARD_64_CHANNELS,
    ConversionStats,
    CatalogResolver,
    compute_fbmae_targets,
    convert_row,
    parse_args,
    run,
    zscore_valid_then_pad,
)


def test_zscore_uses_only_real_samples_then_pads_zero():
    rng = np.random.default_rng(7)
    signal = np.zeros((64, 300), dtype=np.float32)
    signal[0] = rng.normal(10.0, 3.0, size=300)
    mask = np.zeros(64, dtype=np.float32)
    mask[0] = 1.0
    output = zscore_valid_then_pad(signal, mask, valid_samples=300)
    assert abs(float(output[0, :300].mean())) < 1e-5
    assert abs(float(output[0, :300].std()) - 1.0) < 1e-5
    assert np.all(output[0, 300:] == 0)
    assert np.all(output[1:] == 0)


def test_target_shape_finite_and_nonconstant():
    t = np.arange(1000, dtype=np.float32) / 100.0
    waveform = np.zeros((64, 1000), dtype=np.float32)
    waveform[0] = np.sin(2 * np.pi * 10.0 * t)
    mask = np.zeros(64, dtype=np.float32)
    mask[0] = 1.0
    target = compute_fbmae_targets(waveform, mask)
    assert target.shape == (64, 10, 5, 5)
    assert np.isfinite(target).all()
    assert float(target[0].std()) > 0
    assert np.all(target[1:] == 0)


class _Resolver:
    def resolve(self, row):
        return list(STANDARD_64_CHANNELS[:2]), "test"


def test_convert_row_channel_major_and_tail_zero():
    t = np.arange(600, dtype=np.float32) / 200.0
    row = {
        "recording_id": "sub-01/eeg/test.edf",
        "start_seconds": 20.0,
        "valid_samples": 600,
        "sample_rate_hz": 200,
        "channel_count": 2,
        "channel_mask": [True, True],
        "eeg": np.stack(
            [np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 12 * t)], axis=0
        ).tolist(),
    }
    output = convert_row(row, "synthetic", 0, _Resolver(), {}, ConversionStats())
    tokens = np.asarray(output["token_inputs"]).reshape(64, 10, 100)
    channels = np.asarray(output["token_channel_indices"])
    times = np.asarray(output["token_time_indices"])
    assert np.array_equal(channels[:20], np.r_[np.zeros(10), np.ones(10)])
    assert np.array_equal(times[:20], np.r_[np.arange(10), np.arange(10)])
    # 600 @ 200 Hz -> 300 @ 100 Hz; patches 3..9 are temporal zero padding.
    assert np.all(tokens[:2, 3:] == 0)
    assert np.all(tokens[2:] == 0)
    assert len(output["targets"]) == 640 * 5 * 5


def test_end_to_end_writer_reproduces_reference_physical_schema(tmp_path):
    t = np.arange(2000, dtype=np.float32) / 200.0
    source_rows = []
    for row_index in range(2):
        source_rows.append(
            {
                "recording_id": f"sub-{row_index + 1:02d}/eeg/test.edf",
                "start_seconds": 0.0,
                "valid_samples": 2000,
                "sample_rate_hz": 200,
                "channel_count": 2,
                "channel_names": ["Fp1", "Cz"],
                "channel_mask": [True, True],
                "eeg": np.stack(
                    [np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 12 * t)],
                    axis=0,
                ).tolist(),
            }
        )
    source = tmp_path / "source.lance"
    output = tmp_path / "processed"
    lance.write_dataset(pa.Table.from_pylist(source_rows), str(source), mode="create")
    args = parse_args(
        [
            "--input",
            f"synthetic={source}",
            "--output-dir",
            str(output),
            "--read-batch-size",
            "2",
        ]
    )
    run(args)
    actual = lance.dataset(str(output / "synthetic" / "data.lance"))
    assert actual.schema.equals(OUTPUT_SCHEMA, check_metadata=True)
    assert actual.count_rows() == 2
    rows = actual.to_table(columns=["global_idx", "dataset_key"]).to_pylist()
    assert rows == [
        {"global_idx": 0, "dataset_key": "synthetic"},
        {"global_idx": 1, "dataset_key": "synthetic"},
    ]
    summary = json.loads((output / "conversion_summary.json").read_text("utf-8"))
    assert summary["datasets"][0]["status"] == "CONVERTED"
    assert summary["strict_channel_matching"] is False


def test_manifest_writes_datasets_separately_and_indices_restart(tmp_path):
    t = np.arange(2000, dtype=np.float32) / 200.0
    row = {
        "recording_id": "sub-01/eeg/test.edf",
        "start_seconds": 0.0,
        "valid_samples": 2000,
        "sample_rate_hz": 200,
        "channel_count": 1,
        "channel_names": ["Cz"],
        "channel_mask": [True],
        "eeg": np.sin(2 * np.pi * 10 * t)[None, :].tolist(),
    }
    inputs = []
    for key in ("first", "second"):
        source = tmp_path / f"{key}.lance"
        lance.write_dataset(pa.Table.from_pylist([row]), str(source), mode="create")
        inputs.append({"dataset_key": key, "lance_path": str(source)})
    manifest = tmp_path / "datasets.json"
    manifest.write_text(json.dumps({"datasets": inputs}), encoding="utf-8")
    output = tmp_path / "separate"
    run(parse_args(["--manifest", str(manifest), "--output-root", str(output)]))

    for key in ("first", "second"):
        dataset = lance.dataset(str(output / key / "data.lance"))
        values = dataset.to_table(columns=["global_idx", "sample_id", "dataset_key"]).to_pylist()
        assert values == [{"global_idx": 0, "sample_id": 0, "dataset_key": key}]


def test_subset_match_keeps_dataset_and_records_dropped_channel(tmp_path):
    t = np.arange(2000, dtype=np.float32) / 200.0
    row = {
        "recording_id": "sub-01/eeg/test.edf",
        "start_seconds": 0.0,
        "valid_samples": 2000,
        "sample_rate_hz": 200,
        "channel_count": 2,
        "channel_names": ["Cz", "NOT_A_STANDARD_CHANNEL"],
        "channel_mask": [True, True],
        "eeg": np.stack(
            [np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 12 * t)]
        ).tolist(),
    }
    source = tmp_path / "bad.lance"
    output = tmp_path / "rejected"
    lance.write_dataset(pa.Table.from_pylist([row]), str(source), mode="create")
    run(
        parse_args(
            [
                "--input",
                f"bad={source}",
                "--output-root",
                str(output),
            ]
        )
    )
    actual = lance.dataset(str(output / "bad" / "data.lance"))
    row_out = actual.to_table(
        columns=["valid_channel_count", "channel_valid_mask"]
    ).to_pylist()[0]
    assert row_out["valid_channel_count"] == 1
    summary = json.loads((output / "conversion_summary.json").read_text("utf-8"))
    item = summary["datasets"][0]
    assert item["status"] == "CONVERTED"
    assert "NOT_A_STANDARD_CHANNEL" in item["channel_preflight"]["unmapped_channels"]
    assert "NOT_A_STANDARD_CHANNEL" in item["dropped_channels"]


def test_zero_matching_channels_skips_whole_dataset_before_output(tmp_path):
    t = np.arange(2000, dtype=np.float32) / 200.0
    row = {
        "recording_id": "sub-01/eeg/test.edf",
        "start_seconds": 0.0,
        "valid_samples": 2000,
        "sample_rate_hz": 200,
        "channel_count": 1,
        "channel_names": ["NOT_A_STANDARD_CHANNEL"],
        "channel_mask": [True],
        "eeg": np.sin(2 * np.pi * 10 * t)[None, :].tolist(),
    }
    source = tmp_path / "all_bad.lance"
    output = tmp_path / "all_rejected"
    lance.write_dataset(pa.Table.from_pylist([row]), str(source), mode="create")
    run(
        parse_args(
            [
                "--input",
                f"all_bad={source}",
                "--output-root",
                str(output),
            ]
        )
    )
    assert not (output / "all_bad" / "data.lance").exists()
    summary = json.loads((output / "conversion_summary.json").read_text("utf-8"))
    item = summary["datasets"][0]
    assert item["status"] == "SKIPPED_CHANNEL_MISMATCH"
    assert item["channel_preflight"]["zero_mapped_rows"] == 1
