import unittest
from unittest.mock import Mock, patch

from dataset import LanceEEGPretrainDataset as SingleLanceDataset
from configs import Config
from lance_dataset import (
    MultiLanceEEGPretrainDataset,
    _normalize_lance_path,
    _open_lance_dataset,
    load_lance_part_paths,
)


LOCAL_PART = "/tos/mnt/processed_parts/part_00000000_00010000.lance"


class TestMountedLancePaths(unittest.TestCase):
    def test_part_list_keeps_the_mounted_absolute_path(self):
        paths = load_lance_part_paths("part_paths.txt")
        self.assertEqual(paths[0], LOCAL_PART)
        self.assertEqual(len(paths), 923)
        self.assertTrue(all(path.startswith("/tos/mnt/processed_parts/") for path in paths))
        self.assertTrue(all(not path.endswith("/") for path in paths))

    def test_local_open_never_passes_storage_options(self):
        with patch(
            "lance_dataset.lance.dataset",
            return_value=Mock(),
            create=True,
        ) as open_dataset:
            _open_lance_dataset(LOCAL_PART, storage_options={"aws_endpoint": "unused"})

        open_dataset.assert_called_once_with(LOCAL_PART)

    def test_remote_open_keeps_storage_options_for_compatibility(self):
        remote_part = "s3://bucket/prefix/part_00000000_00010000.lance"
        storage_options = {"aws_endpoint": "https://example.invalid"}
        with patch(
            "lance_dataset.lance.dataset",
            return_value=Mock(),
            create=True,
        ) as open_dataset:
            _open_lance_dataset(remote_part, storage_options=storage_options)

        open_dataset.assert_called_once_with(remote_part, storage_options=storage_options)
        self.assertEqual(
            _normalize_lance_path(remote_part),
            remote_part + "/",
        )

    def test_single_and_multi_dataset_use_local_open_without_options(self):
        fake_dataset = Mock()
        fake_dataset.count_rows.return_value = 1
        with patch(
            "dataset.lance.dataset",
            return_value=fake_dataset,
            create=True,
        ) as open_single:
            SingleLanceDataset(LOCAL_PART, storage_options={"aws_endpoint": "unused"})
        open_single.assert_called_once_with(LOCAL_PART)

        multi_dataset = MultiLanceEEGPretrainDataset(
            [LOCAL_PART],
            storage_options={"aws_endpoint": "unused"},
            lengths=[1],
        )
        with patch(
            "lance_dataset.lance.dataset",
            return_value=Mock(),
            create=True,
        ) as open_multi:
            multi_dataset._open_lance_path(LOCAL_PART)
        open_multi.assert_called_once_with(LOCAL_PART)

    def test_multilance_counting_uses_local_open_without_options(self):
        fake_dataset = Mock()
        fake_dataset.count_rows.return_value = 7
        with patch(
            "lance_dataset.lance.dataset",
            return_value=fake_dataset,
            create=True,
        ) as open_dataset:
            multi_dataset = MultiLanceEEGPretrainDataset(
                [LOCAL_PART],
                storage_options={"aws_endpoint": "unused"},
            )

        self.assertEqual(len(multi_dataset), 7)
        open_dataset.assert_called_once_with(LOCAL_PART)

    def test_training_entry_skips_storage_options_for_mounted_parts(self):
        from largeScale_pretrain_main import build_dataset

        with patch("largeScale_pretrain_main.os.path.exists", return_value=True):
            with patch("largeScale_pretrain_main.load_storage_options") as load_options:
                with patch(
                    "largeScale_pretrain_main.MultiLanceEEGPretrainDataset",
                    return_value=Mock(),
                ) as multi_dataset:
                    build_dataset(Config(), rank=1)

        load_options.assert_not_called()
        self.assertIsNone(multi_dataset.call_args.kwargs["storage_options"])

    def test_training_entry_builds_runtime_dataset_from_mounted_lance(self):
        from largeScale_pretrain_main import build_dataset

        cfg = Config()
        cfg.data.input_mode = "canonical_runtime"
        cfg.data.canonical_dataset_key = "egobrain"
        cfg.data.canonical_lance_path = "/opt/ml/omni/egobrain/data.lance"
        cfg.data.runtime_max_samples = 128
        fake_dataset = Mock()
        fake_dataset.valid_samples_audit = {"selected_rows": 128}
        fake_dataset.mapping_audit = {"policy_names": ["egobrain-drop-4-unmatched"]}

        with patch("largeScale_pretrain_main.os.path.exists", return_value=True):
            with patch("largeScale_pretrain_main.load_storage_options") as load_options:
                with patch(
                    "largeScale_pretrain_main.CanonicalRuntimeTargetDataset",
                    return_value=fake_dataset,
                ) as runtime_dataset:
                    result = build_dataset(cfg, rank=1)

        self.assertIs(result, fake_dataset)
        load_options.assert_not_called()
        runtime_dataset.assert_called_once_with(
            lance_path="/opt/ml/omni/egobrain/data.lance",
            dataset_key="egobrain",
            max_samples=128,
            validate=True,
        )


if __name__ == "__main__":
    unittest.main()
