import json
import tempfile
import unittest
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

from canonical_runtime_dataset import CanonicalRuntimeTargetDataset
from runtime_adaptation.canonical import MetadataDrivenCanonicalDecoder
from runtime_adaptation.channel_mapping import CHANNEL_MAPPING_POLICIES
from runtime_adaptation.core import SharedFBMAECore


CATALOGS = {
    "egobrain": (
        "85f920b8a8464acd",
        ("Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9",
         "T7", "CP5", "CP1", "P3", "P7", "PO9", "O1", "Pz", "Oz",
         "O2", "PO10", "P8", "P4", "CP2", "CP6", "T8", "FT10",
         "FC6", "C4", "FC2", "F4", "F8", "Fp2"),
    ),
    "shin2017a": (
        "633f69edb30cb395",
        ("F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8",
         "AFF1h", "AFF2h", "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h",
         "CCP3h", "T7", "P7", "P3", "PPO1h", "POO1", "POO2", "PPO2h",
         "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8"),
    ),
}


def make_schema(dataset_key, map_id, channel_names, source_sfreq):
    payload = {
        "dataset": {"dataset_id": dataset_key},
        "recipe": {"sampling_rates": {"eeg": 100.0}},
        "channel_metadata_catalog": {
            "eeg": [{
                "channel_map_id": map_id,
                "channel_names": list(channel_names),
                "sampling_frequency": source_sfreq,
            }]
        },
    }
    return pa.schema([]).with_metadata({
        b"omni_neuralset.segment_lance": json.dumps(payload).encode("utf-8")
    })


def schema_metadata(dataset_key, map_id, channel_names, source_sfreq):
    return make_schema(dataset_key, map_id, channel_names, source_sfreq).metadata


def make_row(dataset_key, map_id, channel_names):
    times = np.arange(1000, dtype=np.float32) / 100.0
    waveforms = []
    for index in range(len(channel_names)):
        waveform = np.sin(2 * np.pi * (2.0 + 0.2 * index) * times).astype(np.float32)
        waveform = (waveform - waveform.mean()) / waveform.std()
        waveforms.append(waveform)
    eeg = np.stack(waveforms)
    return {
        "segment_id": f"{dataset_key}-segment",
        "dataset_id": dataset_key,
        "source_index": 3,
        "subject": "synthetic",
        "eeg_channel_map_id": map_id,
        "eeg": eeg.reshape(-1).tolist(),
        "eeg_shape": [len(channel_names), 1000],
        "eeg_n_channels": len(channel_names),
        "eeg_channel_indices": list(range(len(channel_names))),
        "valid_channel_mask_json": json.dumps([1] * len(channel_names)),
        "valid_samples": "1000",
    }


class TestMetadataDrivenAdaptation(unittest.TestCase):
    def _adapt(self, dataset_key, source_sfreq):
        map_id, names = CATALOGS[dataset_key]
        decoder = MetadataDrivenCanonicalDecoder.from_schema(
            make_schema(dataset_key, map_id, names, source_sfreq),
            expected_dataset_key=dataset_key,
        )
        window = decoder.decode(make_row(dataset_key, map_id, names))
        return decoder, window, SharedFBMAECore().adapt(window, validate=True)

    def test_egobrain_uses_canonical_rate_and_drops_exactly_four(self):
        decoder, window, sample = self._adapt("egobrain", 256.0)
        self.assertEqual(decoder.source_sfreq, 100.0)
        self.assertEqual(window.sfreq, 100.0)
        self.assertEqual(sample["valid_channel_count"], 28)
        self.assertEqual(sample["dropped_channel_count"], 4)
        self.assertEqual(
            set(sample["dropped_channel_names"].split(",")),
            {"FT9", "PO9", "PO10", "FT10"},
        )
        self.assertEqual(tuple(sample["token_inputs"].shape), (640, 100))
        self.assertEqual(tuple(sample["targets"].shape), (640, 5, 5))

    def test_shin_validation_mode_keeps_12_and_drops_exactly_18(self):
        decoder, window, sample = self._adapt("shin2017a", 200.0)
        expected_dropped = CHANNEL_MAPPING_POLICIES[
            "633f69edb30cb395"
        ].allowed_dropped_channels
        self.assertEqual(decoder.source_sfreq, 100.0)
        self.assertEqual(window.sfreq, 100.0)
        self.assertEqual(sample["valid_channel_count"], 12)
        self.assertEqual(sample["dropped_channel_count"], 18)
        self.assertEqual(
            set(sample["dropped_channel_names"].split(",")), expected_dropped
        )
        self.assertEqual(tuple(sample["token_inputs"].shape), (640, 100))
        self.assertEqual(tuple(sample["targets"].shape), (640, 5, 5))

    def test_unapproved_unknown_channel_is_not_silently_dropped(self):
        map_id, names = CATALOGS["egobrain"]
        changed_names = names[:-1] + ("UNEXPECTED",)
        decoder = MetadataDrivenCanonicalDecoder.from_schema(
            make_schema("egobrain", map_id, changed_names, 256.0),
            expected_dataset_key="egobrain",
        )
        window = decoder.decode(make_row("egobrain", map_id, changed_names))
        with self.assertRaisesRegex(ValueError, "not allowed by mapping policy"):
            SharedFBMAECore().adapt(window)

    def test_schema_dataset_identity_must_match_request(self):
        map_id, names = CATALOGS["egobrain"]
        schema = make_schema("egobrain", map_id, names, 256.0)
        with self.assertRaisesRegex(ValueError, "Dataset identity mismatch"):
            MetadataDrivenCanonicalDecoder.from_schema(
                schema, expected_dataset_key="shin2017a"
            )

    def test_generic_lance_dataset_uses_embedded_catalog(self):
        map_id, names = CATALOGS["egobrain"]
        row = make_row("egobrain", map_id, names)
        table = pa.Table.from_pylist([row]).replace_schema_metadata(
            schema_metadata("egobrain", map_id, names, 256.0)
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "egobrain.lance")
            lance.write_dataset(table, path, mode="overwrite")
            dataset = CanonicalRuntimeTargetDataset(
                path, dataset_key="egobrain", max_samples=1, validate=True
            )
            sample = dataset[0]
        self.assertEqual(dataset.mapping_audit["canonical_sfreq"], 100.0)
        self.assertEqual(sample["valid_channel_count"], 28)
        self.assertEqual(sample["dropped_channel_count"], 4)


if __name__ == "__main__":
    unittest.main()
