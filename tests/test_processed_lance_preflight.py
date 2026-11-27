import numpy as np
import lance
import pyarrow as pa
import pytest

from lance_dataset import preflight_processed_lance_parts


SCHEMA = pa.schema(
    [
        pa.field("num_tokens", pa.int64()),
        pa.field("patch_len", pa.int64()),
        pa.field("n_bands", pa.int64()),
        pa.field("frames_per_patch", pa.int64()),
        pa.field("n_channels", pa.int64()),
        pa.field("token_inputs", pa.list_(pa.float64())),
        pa.field("targets", pa.list_(pa.float64())),
        pa.field("token_channel_indices", pa.list_(pa.int64())),
        pa.field("token_time_indices", pa.list_(pa.int64())),
        pa.field("token_valid_mask", pa.list_(pa.float64())),
        pa.field("channel_valid_mask", pa.list_(pa.float64())),
    ]
)


def make_row(*, inject_nan=False):
    token_inputs = np.zeros((640, 100), dtype=np.float64)
    targets = np.zeros((640, 5, 5), dtype=np.float64)
    channel_mask = np.zeros(64, dtype=np.float64)
    channel_mask[:2] = 1.0
    channel_indices = np.repeat(np.arange(64), 10).astype(np.int64)
    time_indices = np.tile(np.arange(10), 64).astype(np.int64)
    token_mask = channel_mask[channel_indices]
    token_inputs[token_mask == 1] = 0.25
    targets[token_mask == 1] = 0.5
    if inject_nan:
        token_inputs[0, 0] = np.nan
    return {
        "num_tokens": 640,
        "patch_len": 100,
        "n_bands": 5,
        "frames_per_patch": 5,
        "n_channels": 64,
        "token_inputs": token_inputs.reshape(-1).tolist(),
        "targets": targets.reshape(-1).tolist(),
        "token_channel_indices": channel_indices.tolist(),
        "token_time_indices": time_indices.tolist(),
        "token_valid_mask": token_mask.tolist(),
        "channel_valid_mask": channel_mask.tolist(),
    }


def write_part(path, rows):
    lance.write_dataset(pa.Table.from_pylist(rows, schema=SCHEMA), str(path), mode="create")


def test_preflight_accepts_multiple_compatible_parts(tmp_path):
    first = tmp_path / "first.lance"
    second = tmp_path / "second.lance"
    write_part(first, [make_row(), make_row()])
    write_part(second, [make_row()])

    report = preflight_processed_lance_parts(
        [str(first), str(second)], sample_rows_per_part=3
    )

    assert report["parts"] == 2
    assert report["total_rows"] == 3
    assert report["sampled_rows"] == 3
    assert report["lengths"] == [2, 1]
    assert report["expected_shapes"]["token_inputs"] == [640, 100]
    assert report["expected_shapes"]["targets"] == [640, 5, 5]


def test_preflight_rejects_nonfinite_sample(tmp_path):
    bad = tmp_path / "bad.lance"
    write_part(bad, [make_row(inject_nan=True)])

    with pytest.raises(ValueError, match="contains NaN/Inf"):
        preflight_processed_lance_parts([str(bad)], sample_rows_per_part=1)


def test_preflight_rejects_wrong_physical_schema(tmp_path):
    bad = tmp_path / "float32_lists.lance"
    wrong_schema = SCHEMA.set(
        SCHEMA.get_field_index("token_inputs"),
        pa.field("token_inputs", pa.list_(pa.float32())),
    )
    lance.write_dataset(
        pa.Table.from_pylist([make_row()], schema=wrong_schema),
        str(bad),
        mode="create",
    )

    with pytest.raises(ValueError, match="physical schema is incompatible"):
        preflight_processed_lance_parts([str(bad)], sample_rows_per_part=1)
