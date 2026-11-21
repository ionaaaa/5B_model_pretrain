"""Runtime FBMAE Dataset for metadata-described canonical EEG Lance releases."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import lance
from torch.utils.data import Dataset

from runtime_adaptation.canonical import (
    CANONICAL_REQUIRED_COLUMNS,
    MetadataDrivenCanonicalDecoder,
    normalize_integer_scalar,
)
from runtime_adaptation.channel_mapping import CHANNEL_MAPPING_POLICIES
from runtime_adaptation.core import SharedFBMAECore


class CanonicalRuntimeTargetDataset(Dataset):
    """Read full 100 Hz windows and dynamically build common FBMAE tensors."""

    def __init__(
        self,
        lance_path: str,
        dataset_key: str,
        max_samples: Optional[int] = None,
        validate: bool = True,
        index_scan_batch_size: int = 65536,
    ):
        self.lance_path = str(lance_path)
        self.dataset_key = str(dataset_key)
        self.validate = bool(validate)
        self.ds = None

        dataset = self._open_new()
        missing = sorted(set(CANONICAL_REQUIRED_COLUMNS) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"Canonical source Lance is missing columns: {missing}")
        self.decoder = MetadataDrivenCanonicalDecoder.from_schema(
            dataset.schema,
            expected_dataset_key=self.dataset_key,
        )
        self.source_columns = tuple(
            column
            for column in self.decoder.source_columns
            if column in dataset.schema.names
        )
        map_ids = sorted(self.decoder.channel_catalogs)
        policies = []
        for map_id in map_ids:
            policy = CHANNEL_MAPPING_POLICIES.get(map_id)
            if policy is None:
                raise ValueError(
                    f"No explicit FBMAE channel policy for channel_map_id={map_id!r}"
                )
            if policy.dataset_key != self.dataset_key:
                raise ValueError(
                    f"Policy {policy.name!r} does not belong to {self.dataset_key!r}"
                )
            policies.append(policy)
        self.mapping_audit = {
            "dataset_key": self.dataset_key,
            "channel_map_ids": map_ids,
            "policy_names": [policy.name for policy in policies],
            "allowed_dropped_channels": sorted(
                set().union(*(policy.allowed_dropped_channels for policy in policies))
            ),
            "canonical_sfreq": self.decoder.source_sfreq,
        }
        self.source_indices, self.valid_samples_audit = self._find_full_length_rows(
            dataset,
            max_samples=max_samples,
            batch_size=index_scan_batch_size,
        )
        del dataset
        if not self.source_indices:
            audit = self.valid_samples_audit
            raise ValueError(
                f"No {self.dataset_key} rows with valid_samples == 1000 were found. "
                f"scanned_rows={audit['scanned_rows']}, "
                f"null_or_unparseable={audit['null_or_unparseable']}, "
                f"numeric_min={audit['numeric_min']}, "
                f"numeric_max={audit['numeric_max']}, "
                f"top_values={audit['top_values']}, raw_types={audit['raw_types']}"
            )
        self.core = SharedFBMAECore()

    def _open_new(self):
        return lance.dataset(self.lance_path)

    @staticmethod
    def _find_full_length_rows(dataset, max_samples: Optional[int], batch_size: int):
        selected: List[int] = []
        row_offset = 0
        value_counts: Counter = Counter()
        raw_type_counts: Counter = Counter()
        null_or_unparseable = 0
        numeric_min = None
        numeric_max = None
        scanner = dataset.scanner(columns=["valid_samples"], batch_size=batch_size)
        for batch in scanner.to_batches():
            for local_index, raw_value in enumerate(batch.column(0).to_pylist()):
                raw_type_counts[type(raw_value).__name__] += 1
                value = normalize_integer_scalar(raw_value)
                if value is None:
                    null_or_unparseable += 1
                    continue
                value_counts[value] += 1
                numeric_min = value if numeric_min is None else min(numeric_min, value)
                numeric_max = value if numeric_max is None else max(numeric_max, value)
                if value == 1000:
                    selected.append(row_offset + local_index)
            row_offset += batch.num_rows
            if max_samples is not None and len(selected) >= max_samples:
                selected = selected[:max_samples]
                break
        return selected, {
            "scanned_rows": row_offset,
            "null_or_unparseable": null_or_unparseable,
            "numeric_min": numeric_min,
            "numeric_max": numeric_max,
            "top_values": value_counts.most_common(10),
            "raw_types": raw_type_counts.most_common(10),
        }

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["ds"] = None
        return state

    def _open(self):
        if self.ds is None:
            self.ds = self._open_new()
        return self.ds

    def _adapt_row(self, row: Dict):
        window = self.decoder.decode(row)
        return self.core.adapt(window, validate=self.validate)

    def __getitem__(self, index: int):
        source_index = self.source_indices[int(index)]
        row = self._open().take(
            [source_index], columns=list(self.source_columns)
        ).to_pylist()[0]
        return self._adapt_row(row)

    def __getitems__(self, indices: Iterable[int]):
        requested = [int(index) for index in indices]
        if not requested:
            return []
        indexed_sources = [self.source_indices[index] for index in requested]
        # Lance take preserves request order, so no reordering is necessary here.
        rows = self._open().take(
            indexed_sources,
            columns=list(self.source_columns),
        ).to_pylist()
        return [self._adapt_row(row) for row in rows]
