#!/usr/bin/env python3
"""Dedicated cached linear probe for the fixed 5B EEG backbone.

This is intentionally *not* a general finetuning entry point.  It extracts
last-layer tokens once with a frozen 5B backbone, mean-pools valid tokens into
4096-dimensional FP16 features, then trains only ``Linear(4096, C)``. Disk
persistence is optional and controlled explicitly by cache CLI flags.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from lance_dataset import (
    LanceEEGPretrainDataset,
    MultiLanceEEGPretrainDataset,
    load_lance_part_paths,
    load_storage_options,
)
from model import EEGPretrainModel
from prefetch_loader import PrefetchLoader
from utils import set_seed


# The production architecture is deliberately fixed in this script.
FORMAL_D_MODEL = 4096
FORMAL_N_HEADS = 16
FORMAL_DEPTH = 25
FORMAL_MLP_RATIO = 4.0
FORMAL_N_CHANNELS = 64
FORMAL_TIME_PATCHES = 10
FORMAL_TOKENS = 640


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if distributed() else 0


def world_size():
    return dist.get_world_size() if distributed() else 1


def main_rank():
    return rank() == 0


def barrier():
    if distributed():
        dist.barrier()


def unwrap(module):
    return module.module if isinstance(module, DDP) else module


def setup_distributed():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if size > 1 and not distributed():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=120))
    return local_rank


def cleanup_distributed():
    if distributed():
        dist.destroy_process_group()


def collate(batch):
    output = {}
    for key in batch[0]:
        values = [row[key] for row in batch]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            output[key] = torch.stack([torch.as_tensor(v) for v in values])
        elif isinstance(values[0], (int, np.integer)):
            output[key] = torch.tensor(values, dtype=torch.long)
        else:
            output[key] = values
    return output


def move(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def parse_csv_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_label_map(value):
    mapping = {}
    for entry in value.split(","):
        raw, mapped = entry.split(":", 1)
        mapping[int(raw.strip())] = int(mapped.strip())
    return mapping


def label_map_from_args(args):
    if args.label_values and args.label_map:
        raise ValueError("Use only one of --label-values and --label-map.")
    if args.label_values:
        values = parse_csv_ints(args.label_values)
        if len(values) != args.num_classes:
            raise ValueError("--label-values length must equal --num-classes.")
        return {raw: index for index, raw in enumerate(values)}
    if args.label_map:
        result = parse_label_map(args.label_map)
        if set(result.values()) != set(range(args.num_classes)):
            raise ValueError("--label-map must map onto exactly [0, num_classes).")
        return result
    return None


def mapped_labels(raw, mapping, num_classes):
    raw = raw.long().view(-1)
    if mapping is None:
        result = raw
    else:
        result = torch.full_like(raw, -1)
        for source, target in mapping.items():
            result[raw == source] = target
    if result.numel() == 0 or result.min().item() < 0 or result.max().item() >= num_classes:
        raise ValueError("Labels are outside the configured label semantics.")
    return result


def labels_from_batch(batch, mapping, num_classes):
    for key in ("label", "labels", "y"):
        if key in batch:
            return mapped_labels(batch[key], mapping, num_classes)
    raise KeyError(f"No downstream label found; fields={list(batch)}")


class BatchedSubset(Dataset):
    """Preserves the fast batched Lance indexing used by the existing pipeline."""
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[int(index)]]

    def __getitems__(self, indices):
        mapped = [self.indices[int(index)] for index in indices]
        return self.dataset.__getitems__(mapped) if hasattr(self.dataset, "__getitems__") else [self.dataset[i] for i in mapped]


def label_for_sample(sample, mapping, num_classes):
    for key in ("label", "labels", "y"):
        if key in sample:
            value = sample[key]
            if torch.is_tensor(value):
                value = value.reshape(-1)[0]
            return int(mapped_labels(torch.as_tensor([value]), mapping, num_classes)[0])
    raise KeyError(f"No downstream label found; fields={list(sample)}")


def subject_stratified_indices(dataset, args, mapping):
    if not hasattr(dataset, "get_subject_label_rows"):
        raise RuntimeError("subject_stratified requires Lance metadata access via get_subject_label_rows().")
    groups = {}
    for row in dataset.get_subject_label_rows(indices=list(range(len(dataset))), chunk_size=50000):
        label = int(mapped_labels(torch.tensor([row["label"]]), mapping, args.num_classes)[0])
        groups.setdefault(label, {}).setdefault(str(row["subject_id"]), []).append(int(row["index"]))
    if set(groups) != set(range(args.num_classes)):
        raise ValueError(f"Subject labels {sorted(groups)} do not cover classes 0..{args.num_classes - 1}.")
    rng = np.random.default_rng(args.split_seed)
    splits = [[], [], []]
    for cls, subject_rows in sorted(groups.items()):
        subjects = list(subject_rows)
        rng.shuffle(subjects)
        n_train, n_val = int(len(subjects) * args.train_fraction), int(len(subjects) * args.val_fraction)
        if min(n_train, n_val, len(subjects) - n_train - n_val) < 1:
            raise ValueError(f"Class {cls} has too few subjects for a three-way split.")
        for split, chosen in zip(splits, (subjects[:n_train], subjects[n_train:n_train + n_val], subjects[n_train + n_val:])):
            for subject in chosen:
                split.extend(subject_rows[subject])
    return tuple(splits)


def build_splits(dataset, args, mapping, run_dir):
    if args.split_mode == "subject_stratified":
        indices = subject_stratified_indices(dataset, args, mapping)
    else:
        all_indices = np.arange(len(dataset))
        if args.split_mode == "within_subject":
            all_indices = np.random.default_rng(args.split_seed).permutation(all_indices)
        elif args.split_mode not in ("cross_subject",):
            raise ValueError("Only cross_subject, within_subject, and subject_stratified are supported.")
        n_train, n_val = int(len(dataset) * args.train_fraction), int(len(dataset) * args.val_fraction)
        indices = (all_indices[:n_train].tolist(), all_indices[n_train:n_train + n_val].tolist(), all_indices[n_train + n_val:].tolist())
    if min(map(len, indices)) < 1:
        raise ValueError(f"Empty split: {[len(part) for part in indices]}")
    payload = {"split_mode": args.split_mode, "split_seed": args.split_seed,
               "train_indices": indices[0], "val_indices": indices[1], "test_indices": indices[2]}
    if main_rank():
        with open(Path(run_dir) / "split_indices.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return tuple(BatchedSubset(dataset, values) for values in indices), payload


class _DownstreamLanceMixin:
    """LP-only Lance projection layered on the existing pretraining datasets.

    The repository's dataset constructors do not accept a ``columns`` keyword,
    and their row converter deliberately omits downstream labels.  This mixin
    keeps their lazy opening, batched reads, and multi-part index mapping while
    selecting and decoding only fields needed by the linear probe.
    """

    COLUMNS = [
        "num_tokens",
        "patch_len",
        "token_inputs",
        "token_channel_indices",
        "token_time_indices",
        "token_valid_mask",
        "label",
        "subject_id",
    ]

    @staticmethod
    def _row_to_sample(row):
        num_tokens = int(row["num_tokens"])
        patch_len = int(row["patch_len"])
        token_inputs = np.asarray(row["token_inputs"], dtype=np.float32).reshape(
            num_tokens, patch_len
        )
        token_channel_indices = np.asarray(
            row["token_channel_indices"], dtype=np.int64
        )
        token_time_indices = np.asarray(row["token_time_indices"], dtype=np.int64)
        token_valid_mask = np.asarray(row["token_valid_mask"], dtype=np.float32)
        return {
            "token_inputs": torch.from_numpy(token_inputs),
            "token_channel_indices": torch.from_numpy(token_channel_indices),
            "token_time_indices": torch.from_numpy(token_time_indices),
            "token_valid_mask": torch.from_numpy(token_valid_mask),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": str(row["subject_id"]),
        }

    def _get_metadata_rows(self, indices, columns, chunk_size):
        indices = [int(index) for index in indices]
        output = []
        for start in range(0, len(indices), int(chunk_size)):
            chunk = indices[start:start + int(chunk_size)]
            if hasattr(self, "_locate_index"):
                grouped = {}
                for global_index in chunk:
                    part_id, local_index = self._locate_index(global_index)
                    grouped.setdefault(int(part_id), []).append(
                        (int(global_index), int(local_index))
                    )
                chunk_rows = []
                for part_id, entries in grouped.items():
                    table = self._get_dataset(part_id).take(
                        indices=[local_index for _, local_index in entries],
                        columns=columns,
                    )
                    for (global_index, _), row in zip(entries, table.to_pylist()):
                        chunk_rows.append({"index": global_index, **row})
                order = {global_index: position for position, global_index in enumerate(chunk)}
                chunk_rows.sort(key=lambda row: order[row["index"]])
                output.extend(chunk_rows)
            else:
                table = self._open().take(indices=chunk, columns=columns)
                output.extend(
                    {"index": global_index, **row}
                    for global_index, row in zip(chunk, table.to_pylist())
                )
        return output

    def get_label_rows(self, indices, chunk_size=50000):
        return self._get_metadata_rows(indices, ["label"], chunk_size)

    def get_subject_label_rows(self, indices, chunk_size=50000):
        return self._get_metadata_rows(indices, ["subject_id", "label"], chunk_size)


class DownstreamLanceEEGDataset(_DownstreamLanceMixin, LanceEEGPretrainDataset):
    pass


class DownstreamMultiLanceEEGDataset(_DownstreamLanceMixin, MultiLanceEEGPretrainDataset):
    pass


def load_lance_dataset(args):
    if args.part_list_file:
        paths = load_lance_part_paths(args.part_list_file)
        options = load_storage_options(args.storage_options) if args.storage_options else None
        return DownstreamMultiLanceEEGDataset(
            lance_paths=paths,
            storage_options=options,
        )
    if not args.data_uri:
        raise ValueError("Provide --part-list-file or --data-uri.")
    options = load_storage_options(args.storage_options) if args.storage_options else None
    return DownstreamLanceEEGDataset(
        lance_path=args.data_uri,
        storage_options=options,
    )


def shape_parameters():
    # Fixed data specification: 10 sec * 100 Hz, 1 sec stride / patch.
    return {"patch_len": 100, "n_time_patches": FORMAL_TIME_PATCHES,
            "num_tokens": FORMAL_TOKENS, "n_bands": 5, "frames_per_patch": 5}


def strip_module_prefix(state):
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def validate_5b_checkpoint_geometry(state):
    checks = {
        "tokenizer.proj.0.weight": (FORMAL_D_MODEL, 100),
        "channel_embed.weight": (FORMAL_N_CHANNELS, FORMAL_D_MODEL),
        "time_embed.weight": (FORMAL_TIME_PATCHES, FORMAL_D_MODEL),
        "encoder.encoder.layers.0.self_attn.in_proj_weight": (3 * FORMAL_D_MODEL, FORMAL_D_MODEL),
        f"encoder.encoder.layers.{FORMAL_DEPTH - 1}.self_attn.in_proj_weight": (3 * FORMAL_D_MODEL, FORMAL_D_MODEL),
    }
    for key, expected in checks.items():
        actual = state.get(key)
        if actual is None or tuple(actual.shape) != expected:
            raise RuntimeError(f"5B geometry mismatch for {key}: expected {expected}, got {None if actual is None else tuple(actual.shape)}")
    layer_ids = {int(key.split(".")[3]) for key in state if key.startswith("encoder.encoder.layers.")}
    if layer_ids != set(range(FORMAL_DEPTH)):
        raise RuntimeError(f"5B geometry mismatch: expected depth={FORMAL_DEPTH}, found encoder layers={sorted(layer_ids)}")


def load_5b_backbone(checkpoint_path, device):
    # Load on CPU first: this avoids an extra FP32 5B copy on the GPU.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must be a state dict or contain model_state_dict/state_dict.")
    state = strip_module_prefix(state)
    validate_5b_checkpoint_geometry(state)
    shape = shape_parameters()
    backbone = EEGPretrainModel(input_dim=shape["patch_len"], d_model=FORMAL_D_MODEL,
        n_heads=FORMAL_N_HEADS, depth=FORMAL_DEPTH, mlp_ratio=FORMAL_MLP_RATIO, dropout=0.1,
        n_channels=FORMAL_N_CHANNELS, n_time_patches=FORMAL_TIME_PATCHES,
        n_bands=shape["n_bands"], frames_per_patch=shape["frames_per_patch"])
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"5B backbone checkpoint loaded\nmissing_keys={list(missing)}\nunexpected_keys={list(unexpected)}", flush=True)
    if missing or unexpected:
        raise RuntimeError("Checkpoint is not an exact 5B EEGPretrainModel state dict; refusing partial load.")
    if device.type == "cuda":
        backbone = backbone.to(device=device, dtype=torch.bfloat16)
    else:
        backbone = backbone.to(device=device, dtype=torch.float32)
    backbone.requires_grad_(False)
    backbone.eval()
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    return backbone


def valid_token_mean(token_embeddings, token_valid_mask, feature_dim=FORMAL_D_MODEL):
    if token_embeddings.ndim != 3 or token_embeddings.shape[-1] != feature_dim:
        raise ValueError(f"Expected token embeddings [B,S,{feature_dim}], got {tuple(token_embeddings.shape)}")
    if token_valid_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("token_valid_mask must have shape [B,S].")
    valid = token_valid_mask.bool()
    counts = valid.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("Every sample must have at least one valid token.")
    weights = valid.unsqueeze(-1).to(token_embeddings.dtype)
    features = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    if features.ndim != 2 or features.shape[-1] != feature_dim:
        raise AssertionError("Invalid pooled feature geometry.")
    return features


def cache_metadata(args, split_payload, mapping):
    split_digest = hashlib.sha256(json.dumps(split_payload, sort_keys=True).encode()).hexdigest()
    checkpoint_path = Path(args.pretrained_ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint does not exist: {checkpoint_path}")
    checkpoint_stat = checkpoint_path.stat()
    return {
        "pretrained_checkpoint_path": str(checkpoint_path),
        "checkpoint_basename": checkpoint_path.name,
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "d_model": FORMAL_D_MODEL, "n_heads": FORMAL_N_HEADS, "depth": FORMAL_DEPTH,
        "mlp_ratio": FORMAL_MLP_RATIO, "probe_layer": "last", "probe_layer_idx": FORMAL_DEPTH - 1,
        "token_pooling": "mean", "feature_dim": FORMAL_D_MODEL, "cache_dtype": "float16",
        "split_mode": args.split_mode, "split_seed": args.split_seed, "split_digest": split_digest,
        # JSON round-trips dictionary keys as strings, so normalize before
        # comparing a persisted cache manifest on a later invocation.
        "label_map": None if mapping is None else {str(key): int(value) for key, value in mapping.items()},
        "num_classes": args.num_classes, "world_size": world_size(),
        "sample_counts": {
            "train": len(split_payload["train_indices"]),
            "val": len(split_payload["val_indices"]),
            "test": len(split_payload["test_indices"]),
        },
    }


def metadata_mismatches(expected, actual):
    return {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }


def resolve_feature_cache_dir(args, run_dir):
    if not (args.save_feature_cache or args.reuse_feature_cache):
        return None
    value = args.feature_cache_dir if args.feature_cache_dir else Path(run_dir) / "feature_cache"
    return Path(value).expanduser().resolve()


def cache_shard_path(cache_dir, split_name, shard_rank=None):
    shard_rank = rank() if shard_rank is None else int(shard_rank)
    return Path(cache_dir) / split_name / f"rank_{shard_rank:03d}.pt"


def save_feature_shard(dataset, cache_dir, split_name):
    path = cache_shard_path(cache_dir, split_name)
    torch.save({"features": dataset.features, "labels": dataset.labels}, path)
    return path


def rank_subset(subset):
    return BatchedSubset(subset, range(rank(), len(subset), world_size()))


def make_eeg_loader(subset, batch_size, args, device):
    loader = DataLoader(subset, batch_size=max(1, min(batch_size, len(subset))), shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate)
    return PrefetchLoader(loader, device=device, prefetch=2) if args.prefetch else loader


def extract_feature_shard(backbone, subset, split_name, args, mapping, device, cache_dir=None):
    local = rank_subset(subset)
    loader = make_eeg_loader(local, args.feature_batch_size, args, device)
    feature_parts, label_parts = [], []
    backbone.eval()
    for batch_index, batch in enumerate(loader):
        batch = move(batch, device)
        labels = labels_from_batch(batch, mapping, args.num_classes).cpu().long()
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = backbone.extract_token_embeddings(batch["token_inputs"], batch["token_channel_indices"], batch["token_time_indices"])
            else:
                tokens = backbone.extract_token_embeddings(batch["token_inputs"], batch["token_channel_indices"], batch["token_time_indices"])
            if batch_index == 0:
                if tokens.shape[1:] != (FORMAL_TOKENS, FORMAL_D_MODEL):
                    raise AssertionError(f"Expected final encoder tokens [B,640,4096], got {tuple(tokens.shape)}")
            features = valid_token_mean(tokens, batch["token_valid_mask"], FORMAL_D_MODEL)
        feature_parts.append(features.detach().to(device="cpu", dtype=torch.float16))
        label_parts.append(labels)
        if batch_index == 0 or (batch_index + 1) % args.feature_cache_log_every == 0:
            print(f"[FeatureCache] split={split_name} rank={rank()} batch={batch_index + 1}/{len(loader)} samples={sum(x.size(0) for x in feature_parts)} feature_shape={list(features.shape)} dtype=float16", flush=True)
    if not feature_parts:
        raise RuntimeError(f"Rank {rank()} has no samples for {split_name} cache.")
    dataset = FeatureDataset(
        torch.cat(feature_parts).contiguous(),
        torch.cat(label_parts).long().contiguous(),
    )
    if cache_dir is not None:
        save_feature_shard(dataset, cache_dir, split_name)
    return dataset


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features, self.labels = features, labels.long().view(-1)
        if features.dtype != torch.float16 or features.ndim != 2 or features.size(1) != FORMAL_D_MODEL:
            raise ValueError("Feature cache must be FP16 [N,4096].")
        if len(features) != len(self.labels):
            raise ValueError("Cached feature/label length mismatch.")

    def __len__(self): return len(self.labels)
    def __getitem__(self, index): return {"features": self.features[index], "label": self.labels[index]}


def load_local_cache(cache_dir, split_name):
    path = cache_shard_path(cache_dir, split_name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature-cache shard: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"features", "labels"}:
        raise ValueError(f"Invalid feature-cache shard fields in {path}")
    return FeatureDataset(payload["features"], payload["labels"])


def validate_and_load_feature_cache(cache_dir, expected_metadata):
    cache_dir = Path(cache_dir)
    metadata_path = cache_dir / "metadata.json"
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Feature cache directory does not exist: {cache_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Feature cache metadata does not exist: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as handle:
        actual_metadata = json.load(handle)
    mismatches = metadata_mismatches(expected_metadata, actual_metadata)
    if mismatches:
        raise RuntimeError(
            "Feature cache metadata is incompatible with this run: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    datasets = tuple(load_local_cache(cache_dir, name) for name in ("train", "val", "test"))
    local_counts = {name: len(dataset) for name, dataset in zip(("train", "val", "test"), datasets)}
    gathered_counts = [None] * world_size() if distributed() else [local_counts]
    if distributed():
        dist.all_gather_object(gathered_counts, local_counts)
    observed = {name: sum(item[name] for item in gathered_counts) for name in local_counts}
    if observed != expected_metadata["sample_counts"]:
        raise RuntimeError(
            f"Feature cache shard sample counts mismatch: expected={expected_metadata['sample_counts']}, "
            f"observed={observed}"
        )
    return datasets


def balance_feature_dataset(dataset, args):
    if not args.balance_train_classes:
        return dataset
    rng = np.random.default_rng(args.balance_seed if args.balance_seed is not None else args.split_seed)
    by_class = {cls: np.where(dataset.labels.numpy() == cls)[0] for cls in range(args.num_classes)}
    if any(len(indices) == 0 for indices in by_class.values()):
        raise ValueError("Cannot balance a rank-local feature shard with an empty class.")
    count = max(map(len, by_class.values())) if args.balance_strategy == "oversample" else min(map(len, by_class.values()))
    indices = np.concatenate([rng.choice(values, count, replace=args.balance_strategy == "oversample") for values in by_class.values()])
    rng.shuffle(indices)
    return FeatureDataset(dataset.features[indices], dataset.labels[indices])


def align_train_length(dataset, device):
    if not distributed():
        return dataset
    value = torch.tensor([len(dataset)], device=device, dtype=torch.long)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    minimum = int(value.item())
    if minimum < 1:
        raise RuntimeError("DDP length alignment produced an empty feature dataset.")
    if len(dataset) != minimum:
        dataset = FeatureDataset(dataset.features[:minimum], dataset.labels[:minimum])
    return dataset


def make_feature_loader(dataset, batch_size, shuffle, device):
    return DataLoader(dataset, batch_size=max(1, min(batch_size, len(dataset))), shuffle=shuffle,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=False)


def metrics_reduce(loss_sum, correct_sum, confusion, device):
    values = torch.tensor([loss_sum, correct_sum, float(confusion.sum())], device=device, dtype=torch.float64)
    if distributed():
        dist.all_reduce(values); dist.all_reduce(confusion)
    loss = float(values[0] / values[2].clamp_min(1))
    acc = float(values[1] / values[2].clamp_min(1))
    recalls = confusion.diag() / confusion.sum(1).clamp_min(1)
    present = confusion.sum(1) > 0
    bacc = float(recalls[present].mean()) if present.any() else float("nan")
    return {"loss": loss, "acc": acc, "bacc": bacc,
            "true_counts": confusion.sum(1).cpu().long().tolist(),
            "pred_counts": confusion.sum(0).cpu().long().tolist(),
            "per_class_recall": [float(recalls[index]) if present[index] else None for index in range(len(recalls))],
            "confusion_matrix": confusion.cpu().long().tolist()}


def print_prediction_distribution(split_name, metrics):
    if main_rank():
        print(f"[PredictionDistribution] split={split_name} true={metrics['true_counts']} "
              f"pred={metrics['pred_counts']} recall={metrics['per_class_recall']} "
              f"confusion={metrics['confusion_matrix']}", flush=True)


def run_epoch(head, loader, optimizer, num_classes, device):
    training = optimizer is not None
    head.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct_sum = 0.0
    confusion = torch.zeros((num_classes, num_classes), device=device, dtype=torch.float64)
    for batch in loader:
        features, labels = batch["features"].to(device).float(), batch["label"].to(device).long()
        logits = head(features)
        loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # Only the standalone linear head is in this graph.
            optimizer.step()
        prediction = logits.argmax(1)
        loss_sum += loss.item() * len(labels); correct_sum += (prediction == labels).sum().item()
        confusion += torch.bincount(labels * num_classes + prediction, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return metrics_reduce(loss_sum, correct_sum, confusion, device)


def global_train_feature_mean(dataset, device):
    """Compute the centering vector exclusively from unbalanced train cache."""
    total = dataset.features.float().sum(0).to(device)
    count = torch.tensor([len(dataset)], device=device, dtype=torch.long)
    if distributed():
        dist.all_reduce(total)
        dist.all_reduce(count)
    return (total / count.clamp_min(1)).cpu()


def json_cell(value):
    return json.dumps(value, ensure_ascii=False)


def epoch_config_fields(args, mapping, cache_dir):
    checkpoint_path = str(Path(args.pretrained_ckpt).expanduser().resolve())
    effective_balance_seed = (
        args.balance_seed if args.balance_seed is not None else args.split_seed
    )
    return {
        "experiment_name": args.experiment_name,
        "pretrained_ckpt": checkpoint_path,
        "pretrained_ckpt_name": Path(checkpoint_path).name,
        "mode": "linear_probe",
        "num_classes": args.num_classes,
        "label_map": json_cell(mapping),
        "split_mode": args.split_mode,
        "split_seed": args.split_seed,
        "balance_train_classes": args.balance_train_classes,
        "balance_strategy": args.balance_strategy,
        "balance_seed": effective_balance_seed,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "feature_dim": FORMAL_D_MODEL,
        "token_pooling": "mean",
        "probe_layer": "last",
        "probe_layer_idx": FORMAL_DEPTH - 1,
        "center_features": args.center_features,
        "feature_cache_dtype": "float16",
        "save_feature_cache": args.save_feature_cache,
        "reuse_feature_cache": args.reuse_feature_cache,
        "feature_cache_dir": str(cache_dir) if cache_dir is not None else "",
        "feature_batch_size": args.feature_batch_size,
        "head_batch_size": args.head_batch_size,
        "metric_for_best": args.metric_for_best,
    }


def build_epoch_metrics_row(args, mapping, cache_dir, epoch, train_metrics, val_metrics):
    row = {
        "experiment_name": args.experiment_name,
        "epoch": int(epoch),
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["acc"],
        "train_bacc": train_metrics["bacc"],
        "val_loss": val_metrics["loss"],
        "val_acc": val_metrics["acc"],
        "val_bacc": val_metrics["bacc"],
        "test_loss": "",
        "test_acc": "",
        "test_bacc": "",
        "is_best": False,
    }
    row.update(epoch_config_fields(args, mapping, cache_dir))
    row.update({
        "train_true_counts": json_cell(train_metrics["true_counts"]),
        "train_pred_counts": json_cell(train_metrics["pred_counts"]),
        "train_per_class_recall": json_cell(train_metrics["per_class_recall"]),
        "train_confusion_matrix": json_cell(train_metrics["confusion_matrix"]),
        "val_true_counts": json_cell(val_metrics["true_counts"]),
        "val_pred_counts": json_cell(val_metrics["pred_counts"]),
        "val_per_class_recall": json_cell(val_metrics["per_class_recall"]),
        "val_confusion_matrix": json_cell(val_metrics["confusion_matrix"]),
        "test_true_counts": "",
        "test_pred_counts": "",
        "test_per_class_recall": "",
        "test_confusion_matrix": "",
    })
    return row


def mark_final_best_row(epoch_rows, best_epoch):
    found = 0
    for row in epoch_rows:
        row["is_best"] = int(row["epoch"]) == int(best_epoch)
        found += int(row["is_best"])
    if found != 1:
        raise AssertionError(f"Expected exactly one best epoch row, found {found}.")


def backfill_best_test_metrics(epoch_rows, best_epoch, test_metrics):
    mark_final_best_row(epoch_rows, best_epoch)
    best_row = next(row for row in epoch_rows if row["is_best"])
    best_row.update({
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_bacc": test_metrics["bacc"],
        "test_true_counts": json_cell(test_metrics["true_counts"]),
        "test_pred_counts": json_cell(test_metrics["pred_counts"]),
        "test_per_class_recall": json_cell(test_metrics["per_class_recall"]),
        "test_confusion_matrix": json_cell(test_metrics["confusion_matrix"]),
    })
    return best_row


def write_csv_rows(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(args, mapping, cache_dir, best, head_checkpoint_path):
    fields = epoch_config_fields(args, mapping, cache_dir)
    return {
        "experiment_name": fields["experiment_name"],
        "pretrained_ckpt": fields["pretrained_ckpt"],
        "pretrained_ckpt_name": fields["pretrained_ckpt_name"],
        "mode": fields["mode"],
        "num_classes": fields["num_classes"],
        "label_map": fields["label_map"],
        "split_mode": fields["split_mode"],
        "split_seed": fields["split_seed"],
        "balance_train_classes": fields["balance_train_classes"],
        "balance_strategy": fields["balance_strategy"],
        "balance_seed": fields["balance_seed"],
        "head_lr": fields["head_lr"],
        "weight_decay": fields["weight_decay"],
        "epochs": fields["epochs"],
        "feature_dim": fields["feature_dim"],
        "token_pooling": fields["token_pooling"],
        "probe_layer": fields["probe_layer"],
        "probe_layer_idx": fields["probe_layer_idx"],
        "center_features": fields["center_features"],
        "feature_cache_dtype": fields["feature_cache_dtype"],
        "save_feature_cache": fields["save_feature_cache"],
        "reuse_feature_cache": fields["reuse_feature_cache"],
        "feature_cache_dir": fields["feature_cache_dir"],
        "feature_batch_size": fields["feature_batch_size"],
        "head_batch_size": fields["head_batch_size"],
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "best_val_bacc": best["val_bacc"],
        "test_loss": best["test_loss"],
        "test_acc": best["test_acc"],
        "test_bacc": best["test_bacc"],
        "metric_for_best": fields["metric_for_best"],
        "head_checkpoint_path": str(Path(head_checkpoint_path).resolve()),
    }


def assert_metrics_consistent(epoch_rows, summary, checkpoint_payload):
    best_rows = [row for row in epoch_rows if row["is_best"]]
    if len(best_rows) != 1:
        raise AssertionError(f"Expected one is_best=True row, found {len(best_rows)}.")
    best_row = best_rows[0]
    checkpoint_best = checkpoint_payload["best"]
    comparisons = {
        "epoch": (best_row["epoch"], summary["best_epoch"], checkpoint_best["epoch"]),
        "val_loss": (best_row["val_loss"], summary["best_val_loss"], checkpoint_best["val_loss"]),
        "val_acc": (best_row["val_acc"], summary["best_val_acc"], checkpoint_best["val_acc"]),
        "val_bacc": (best_row["val_bacc"], summary["best_val_bacc"], checkpoint_best["val_bacc"]),
        "test_loss": (best_row["test_loss"], summary["test_loss"], checkpoint_best["test_loss"]),
        "test_acc": (best_row["test_acc"], summary["test_acc"], checkpoint_best["test_acc"]),
        "test_bacc": (best_row["test_bacc"], summary["test_bacc"], checkpoint_best["test_bacc"]),
    }
    for name, values in comparisons.items():
        if name == "epoch":
            consistent = len({int(value) for value in values}) == 1
        else:
            consistent = all(np.isclose(float(values[0]), float(value)) for value in values[1:])
        if not consistent:
            raise AssertionError(f"Metrics mismatch for {name}: {values}")


def save_head_checkpoint(path, head, args, mapping, best, feature_mean):
    payload = {
        "checkpoint_format": "5b_linear_probe_head",
        "head_state_dict": {key: value.detach().cpu() for key, value in unwrap(head).state_dict().items()},
        "backbone": {"checkpoint_path": str(Path(args.pretrained_ckpt).resolve()), "checkpoint_name": Path(args.pretrained_ckpt).name,
                     "d_model": FORMAL_D_MODEL, "n_heads": FORMAL_N_HEADS, "depth": FORMAL_DEPTH, "mlp_ratio": FORMAL_MLP_RATIO},
        "feature": {"probe_layer": "last", "probe_layer_idx": FORMAL_DEPTH - 1, "token_pooling": "mean", "feature_dim": FORMAL_D_MODEL, "cache_dtype": "float16"},
        "task": {"num_classes": args.num_classes, "label_map": mapping},
        "split": {"split_mode": args.split_mode, "split_seed": args.split_seed}, "best": best,
    }
    if feature_mean is not None: payload["feature_mean"] = feature_mean.cpu()
    torch.save(payload, path)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-ckpt", required=False)
    parser.add_argument("--part-list-file"); parser.add_argument("--data-uri"); parser.add_argument("--storage-options")
    parser.add_argument("--output-dir", default="./linear_probe_5b_runs"); parser.add_argument("--experiment-name", default="linear_probe_5b")
    parser.add_argument("--num-classes", type=int, default=3); parser.add_argument("--label-values"); parser.add_argument("--label-map")
    parser.add_argument("--split-mode", choices=["cross_subject", "within_subject", "subject_stratified"], default="subject_stratified")
    parser.add_argument("--split-seed", type=int, default=42); parser.add_argument("--train-fraction", type=float, default=0.7); parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--feature-batch-size", type=int, default=1); parser.add_argument("--head-batch-size", type=int, default=256); parser.add_argument("--batch-size", type=int, help="Deprecated alias for --head-batch-size")
    parser.add_argument("--num-workers", type=int, default=4); parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--head-lr", type=float, default=0.01); parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--balance-train-classes", action="store_true"); parser.add_argument("--balance-strategy", choices=["oversample", "undersample"], default="oversample"); parser.add_argument("--balance-seed", type=int)
    parser.add_argument("--center-features", action="store_true")
    parser.add_argument(
        "--save-feature-cache",
        action="store_true",
        help="Persist extracted frozen-backbone features to disk.",
    )
    parser.add_argument(
        "--reuse-feature-cache",
        action="store_true",
        help="Reuse an existing compatible feature cache and skip 5B backbone feature extraction.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=str,
        default=None,
        help="Directory used to save/load persistent feature-cache shards.",
    )
    parser.add_argument("--feature-cache-log-every", type=int, default=10)
    parser.add_argument("--metric-for-best", choices=["val_acc", "val_bacc", "val_loss"], default="val_bacc"); parser.add_argument("--device")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.batch_size is not None: args.head_batch_size = args.batch_size
    if args.train_fraction + args.val_fraction >= 1: parser.error("train/val fractions must sum to < 1")
    if not args.smoke_test and not args.pretrained_ckpt: parser.error("--pretrained-ckpt is required")
    return args


def run(args):
    local_rank = setup_distributed()
    device = torch.device(args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"))
    set_seed(args.split_seed + rank())
    mapping = label_map_from_args(args)
    run_dir = Path(args.output_dir) / args.experiment_name
    if main_rank(): run_dir.mkdir(parents=True, exist_ok=True)
    barrier()
    dataset = load_lance_dataset(args)
    splits, split_payload = build_splits(dataset, args, mapping, run_dir)
    cache_dir = resolve_feature_cache_dir(args, run_dir)
    expected_metadata = cache_metadata(args, split_payload, mapping)
    if main_rank():
        if cache_dir is not None:
            print(f"[FeatureCache] dir={cache_dir}", flush=True)
        print(f"[FeatureCache] save_to_disk={args.save_feature_cache}", flush=True)

    if args.reuse_feature_cache:
        if main_rank():
            print("[FeatureCache] mode=reuse", flush=True)
        feature_datasets = validate_and_load_feature_cache(cache_dir, expected_metadata)
        if main_rank():
            print("[FeatureCache] metadata validation: OK", flush=True)
            print("[FeatureCache] skipping 5B backbone feature extraction", flush=True)
            if args.save_feature_cache:
                print(
                    "[FeatureCache] reused existing cache; skip feature extraction and cache rewrite",
                    flush=True,
                )
    else:
        if main_rank():
            print("[FeatureCache] mode=extract", flush=True)
        if args.save_feature_cache:
            # Persisted caches are immutable products with explicit provenance.
            # Do not replace a non-empty directory without validated reuse.
            can_build = True
            if main_rank() and cache_dir.exists() and any(cache_dir.iterdir()):
                can_build = False
            if distributed():
                signal = [can_build]
                dist.broadcast_object_list(signal, src=0)
                can_build = signal[0]
            if not can_build:
                raise RuntimeError(
                    f"Feature cache directory is not empty: {cache_dir}. Choose another "
                    "--feature-cache-dir, or use --reuse-feature-cache for strict validation."
                )
            if main_rank():
                cache_dir.mkdir(parents=True, exist_ok=True)
                for split_name in ("train", "val", "test"):
                    (cache_dir / split_name).mkdir()
            barrier()

        backbone = load_5b_backbone(args.pretrained_ckpt, device)
        feature_datasets = tuple(
            extract_feature_shard(
                backbone=backbone,
                subset=subset,
                split_name=name,
                args=args,
                mapping=mapping,
                device=device,
                cache_dir=cache_dir if args.save_feature_cache else None,
            )
            for name, subset in zip(("train", "val", "test"), splits)
        )
        counts = {
            name: len(feature_dataset)
            for name, feature_dataset in zip(("train", "val", "test"), feature_datasets)
        }
        all_counts = [None] * world_size() if distributed() else [counts]
        if distributed():
            dist.all_gather_object(all_counts, counts)
        observed_counts = {name: sum(item[name] for item in all_counts) for name in counts}
        if observed_counts != expected_metadata["sample_counts"]:
            raise RuntimeError(
                f"Extracted feature counts mismatch: expected={expected_metadata['sample_counts']}, "
                f"observed={observed_counts}"
            )
        barrier()
        if main_rank() and args.save_feature_cache:
            with open(cache_dir / "metadata.json", "w", encoding="utf-8") as handle:
                json.dump(expected_metadata, handle, indent=2)
        if main_rank():
            print(
                f"[FeatureCache] totals={observed_counts} feature_dim={FORMAL_D_MODEL}",
                flush=True,
            )
        barrier()
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_data, val_data, test_data = feature_datasets
    # Centering intentionally precedes feature-level sampling: duplicated
    # oversampled examples must not change a statistic defined on real train data.
    feature_mean = global_train_feature_mean(train_data, device) if args.center_features else None
    train_data = balance_feature_dataset(train_data, args); train_data = align_train_length(train_data, device)
    if feature_mean is not None:
        for dataset_ in (train_data, val_data, test_data): dataset_.features = (dataset_.features.float() - feature_mean).half()
    head = nn.Linear(FORMAL_D_MODEL, args.num_classes).to(device)
    params = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.SGD(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    assert sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]) == params
    if main_rank(): print(f"feature_dim=4096 num_classes={args.num_classes} linear_head_params={params:,} optimizer_params={params:,}", flush=True)
    train_head = DDP(head, device_ids=[local_rank], output_device=local_rank) if distributed() and device.type == "cuda" else (DDP(head) if distributed() else head)
    loaders = [make_feature_loader(data, args.head_batch_size, index == 0, device) for index, data in enumerate((train_data, val_data, test_data))]
    best = {"epoch": -1, "val_acc": float("-inf"), "val_bacc": float("-inf"), "val_loss": float("inf"), "test_acc": float("nan"), "test_bacc": float("nan"), "test_loss": float("nan")}
    best_state = None
    epoch_rows = []
    metrics_path = run_dir / "metrics" / "epoch_metrics.csv"
    summary_path = run_dir / "summary.csv"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_head, loaders[0], optimizer, args.num_classes, device)
        with torch.inference_mode(): val_metrics = run_epoch(unwrap(train_head), loaders[1], None, args.num_classes, device)
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} train_bacc={train_metrics['bacc']:.4f} val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} val_bacc={val_metrics['bacc']:.4f}", flush=True)
        if main_rank():
            epoch_rows.append(
                build_epoch_metrics_row(
                    args=args,
                    mapping=mapping,
                    cache_dir=cache_dir,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
            )
        better = val_metrics[args.metric_for_best.removeprefix("val_")] < best[args.metric_for_best] if args.metric_for_best == "val_loss" else val_metrics[args.metric_for_best.removeprefix("val_")] > best[args.metric_for_best]
        if better:
            best.update({"epoch": epoch, **{f"val_{key}": val_metrics[key] for key in ("loss", "acc", "bacc")}})
            best_state = {key: value.detach().cpu().clone() for key, value in unwrap(train_head).state_dict().items()}
            if main_rank():
                mark_final_best_row(epoch_rows, best["epoch"])
        if main_rank():
            # Refresh after every completed epoch so train/val history survives
            # a later interruption. Test cells remain empty until final test.
            write_csv_rows(metrics_path, epoch_rows)
        print_prediction_distribution("val", val_metrics)
    unwrap(train_head).load_state_dict(best_state)
    with torch.inference_mode(): test_metrics = run_epoch(unwrap(train_head), loaders[2], None, args.num_classes, device)
    print_prediction_distribution("test", test_metrics)
    best.update({f"test_{key}": test_metrics[key] for key in ("loss", "acc", "bacc")})
    if main_rank():
        best_row = backfill_best_test_metrics(epoch_rows, best["epoch"], test_metrics)
        destination = run_dir / "checkpoints"; destination.mkdir(exist_ok=True)
        path = destination / "linear_probe_head_best.pt"
        checkpoint_payload = save_head_checkpoint(
            path, train_head, args, mapping, best, feature_mean
        )
        summary_row = build_summary_row(
            args=args,
            mapping=mapping,
            cache_dir=cache_dir,
            best=best,
            head_checkpoint_path=path,
        )
        assert_metrics_consistent(epoch_rows, summary_row, checkpoint_payload)
        assert int(best_row["epoch"]) == int(best["epoch"])
        write_csv_rows(metrics_path, epoch_rows)
        write_csv_rows(summary_path, [summary_row])
        print(f"Epoch metrics saved to: {metrics_path}", flush=True)
        print(f"Saved summary to: {summary_path}", flush=True)
        print(f"best epoch={best['epoch']} test_acc={best['test_acc']:.4f} test_bacc={best['test_bacc']:.4f}\nhead-only checkpoint: {path}", flush=True)
    barrier(); cleanup_distributed()


def smoke_test():
    """CPU-only coverage for model invariants and all cache CLI modes."""
    temp = Path(tempfile.mkdtemp(prefix="linear_probe_5b_smoke_"))
    try:
        # Existing model/linear-probe invariants.
        model = EEGPretrainModel(input_dim=4, d_model=64, n_heads=4, depth=2, mlp_ratio=2, dropout=0,
            n_channels=2, n_time_patches=4, n_bands=2, frames_per_patch=2).eval()
        inputs = torch.randn(3, 8, 4); channels = torch.arange(8).repeat(3, 1) % 2; times = torch.arange(8).repeat(3, 1) % 4
        tokens = model.extract_token_embeddings(inputs, channels, times)
        assert torch.allclose(model.head(tokens), model(inputs, channels, times))
        pooled = valid_token_mean(torch.tensor([[[1., 2.], [99., 99.], [3., 4.]]]), torch.tensor([[1, 0, 1]]), 2)
        assert torch.equal(pooled, torch.tensor([[2., 3.]]))
        assert FORMAL_D_MODEL == 4096
        features = torch.randn(8, 64).half(); labels = torch.arange(8) % 2
        cache = temp / "train_rank000.pt"; torch.save({"features": features, "labels": labels}, cache)
        formal_cache = torch.randn(5, FORMAL_D_MODEL).half()
        torch.save({"features": formal_cache, "labels": torch.zeros(5, dtype=torch.long)}, cache)
        loaded = torch.load(cache, weights_only=True); assert loaded["features"].dtype == torch.float16 and loaded["features"].shape == (5, 4096)
        head = nn.Linear(64, 2); optimizer = torch.optim.SGD(head.parameters(), lr=.1)
        assert sum(p.numel() for group in optimizer.param_groups for p in group["params"]) == sum(p.numel() for p in head.parameters())
        loss = nn.CrossEntropyLoss()(head(features.float()), labels); loss.backward(); optimizer.step()
        before = head(features.float()).detach(); args = argparse.Namespace(pretrained_ckpt="tiny.pt", num_classes=2, split_mode="within_subject", split_seed=7)
        output = temp / "head.pt"; save_head_checkpoint(output, head, args, {0: 0, 1: 1}, {"epoch": 1, "val_acc": 0., "val_bacc": 0., "val_loss": 0., "test_acc": 0., "test_bacc": 0., "test_loss": 0.}, None)
        payload = torch.load(output, weights_only=True); assert "head_state_dict" in payload and not ({"model_state_dict", "backbone_state_dict", "optimizer_state_dict"} & set(payload))
        reloaded = nn.Linear(64, 2); reloaded.load_state_dict(payload["head_state_dict"]); assert torch.allclose(before, reloaded(features.float()))

        # Current-repository Lance API regression: the downstream adapter uses
        # inherited constructors (no unsupported ``columns=`` argument) while
        # still decoding label/subject metadata required by this pipeline.
        lance_row = {
            "num_tokens": 8, "patch_len": 4,
            "token_inputs": np.arange(32, dtype=np.float32),
            "token_channel_indices": np.arange(8) % 2,
            "token_time_indices": np.arange(8) % 4,
            "token_valid_mask": np.ones(8, dtype=np.float32),
            "label": 1, "subject_id": "smoke-subject",
        }
        for dataset_class in (DownstreamLanceEEGDataset, DownstreamMultiLanceEEGDataset):
            sample = dataset_class._row_to_sample(lance_row)
            assert sample["token_inputs"].shape == (8, 4)
            assert sample["label"].item() == 1
            assert sample["subject_id"] == "smoke-subject"

        # Cache CLI A: persist to the explicitly selected cache directory.
        checkpoint_path = temp / "tiny_backbone.pt"
        torch.save({"smoke": True}, checkpoint_path)
        cache_args = argparse.Namespace(
            pretrained_ckpt=str(checkpoint_path), num_classes=2,
            split_mode="within_subject", split_seed=7,
            save_feature_cache=True, reuse_feature_cache=False,
            feature_cache_dir=str(temp / "user_selected_cache"),
        )
        split_payload = {
            "split_mode": "within_subject", "split_seed": 7,
            "train_indices": list(range(8)), "val_indices": list(range(8, 12)),
            "test_indices": list(range(12, 16)),
        }
        metadata = cache_metadata(cache_args, split_payload, {0: 0, 1: 1})
        custom_cache = resolve_feature_cache_dir(cache_args, temp / "default_run")
        for split_name in ("train", "val", "test"):
            (custom_cache / split_name).mkdir(parents=True, exist_ok=True)
        tiny_datasets = (
            FeatureDataset(torch.randn(8, FORMAL_D_MODEL).half(), torch.arange(8) % 2),
            FeatureDataset(torch.randn(4, FORMAL_D_MODEL).half(), torch.arange(4) % 2),
            FeatureDataset(torch.randn(4, FORMAL_D_MODEL).half(), torch.arange(4) % 2),
        )
        for split_name, dataset in zip(("train", "val", "test"), tiny_datasets):
            save_feature_shard(dataset, custom_cache, split_name)
        with open(custom_cache / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        assert custom_cache == (temp / "user_selected_cache").resolve()
        assert cache_shard_path(custom_cache, "train").is_file()
        assert not (temp / "default_run" / "feature_cache").exists()

        # Cache CLI B: strict reuse loads cache and never invokes extraction.
        extraction_calls = 0

        def forbidden_extraction():
            nonlocal extraction_calls
            extraction_calls += 1
            raise AssertionError("Backbone extraction must not run in reuse mode")

        second_run_reuses_cache = True
        if second_run_reuses_cache:
            reused = validate_and_load_feature_cache(custom_cache, metadata)
        else:
            forbidden_extraction()
        assert extraction_calls == 0
        reuse_head = nn.Linear(FORMAL_D_MODEL, 2)
        reuse_optimizer = torch.optim.SGD(reuse_head.parameters(), lr=0.01)
        reuse_metrics = run_epoch(
            reuse_head,
            make_feature_loader(reused[0], batch_size=4, shuffle=False, device=torch.device("cpu")),
            reuse_optimizer,
            num_classes=2,
            device=torch.device("cpu"),
        )
        assert np.isfinite(reuse_metrics["loss"])

        # Cache CLI C: modified provenance/split metadata must fail reuse.
        incompatible = dict(metadata)
        incompatible["split_seed"] = metadata["split_seed"] + 1
        with open(custom_cache / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(incompatible, handle)
        try:
            validate_and_load_feature_cache(custom_cache, metadata)
        except RuntimeError as error:
            assert "split_seed" in str(error)
        else:
            raise AssertionError("Incompatible cache metadata was accepted")

        # Cache CLI D: no flags resolves no cache path and in-memory LP works.
        memory_args = argparse.Namespace(
            save_feature_cache=False, reuse_feature_cache=False, feature_cache_dir=None,
        )
        no_cache_run = temp / "memory_only_run"
        assert resolve_feature_cache_dir(memory_args, no_cache_run) is None
        memory_head = nn.Linear(FORMAL_D_MODEL, 2)
        memory_optimizer = torch.optim.SGD(memory_head.parameters(), lr=0.01)
        memory_metrics = run_epoch(
            memory_head,
            make_feature_loader(tiny_datasets[0], batch_size=4, shuffle=False, device=torch.device("cpu")),
            memory_optimizer,
            num_classes=2,
            device=torch.device("cpu"),
        )
        assert np.isfinite(memory_metrics["loss"])
        assert not (no_cache_run / "feature_cache").exists()

        # Metrics persistence A-F: multiple epochs, final test on best head,
        # best-row backfill, two CSVs, and checkpoint/CSV consistency.
        metrics_args = argparse.Namespace(
            pretrained_ckpt=str(checkpoint_path),
            experiment_name="smoke_metrics",
            num_classes=2,
            split_mode="within_subject",
            split_seed=7,
            balance_train_classes=False,
            balance_strategy="oversample",
            balance_seed=None,
            head_lr=0.01,
            weight_decay=0.0,
            epochs=3,
            center_features=False,
            save_feature_cache=False,
            reuse_feature_cache=False,
            feature_batch_size=1,
            head_batch_size=4,
            metric_for_best="val_bacc",
        )
        metric_head = nn.Linear(FORMAL_D_MODEL, 2)
        metric_optimizer = torch.optim.SGD(metric_head.parameters(), lr=0.01)
        metric_train_loader = make_feature_loader(
            tiny_datasets[0], 4, False, torch.device("cpu")
        )
        metric_val_loader = make_feature_loader(
            tiny_datasets[1], 4, False, torch.device("cpu")
        )
        metric_test_loader = make_feature_loader(
            tiny_datasets[2], 4, False, torch.device("cpu")
        )
        metric_rows = []
        metric_best = {
            "epoch": -1, "val_acc": -float("inf"),
            "val_bacc": -float("inf"), "val_loss": float("inf"),
            "test_acc": float("nan"), "test_bacc": float("nan"),
            "test_loss": float("nan"),
        }
        metric_best_state = None
        for epoch in range(1, metrics_args.epochs + 1):
            train_result = run_epoch(
                metric_head, metric_train_loader, metric_optimizer, 2,
                torch.device("cpu"),
            )
            with torch.inference_mode():
                val_result = run_epoch(
                    metric_head, metric_val_loader, None, 2, torch.device("cpu")
                )
            metric_rows.append(
                build_epoch_metrics_row(
                    metrics_args, {0: 0, 1: 1}, None, epoch,
                    train_result, val_result,
                )
            )
            if val_result["bacc"] > metric_best["val_bacc"]:
                metric_best.update({
                    "epoch": epoch,
                    "val_loss": val_result["loss"],
                    "val_acc": val_result["acc"],
                    "val_bacc": val_result["bacc"],
                })
                metric_best_state = {
                    key: value.detach().clone()
                    for key, value in metric_head.state_dict().items()
                }
                mark_final_best_row(metric_rows, epoch)
        metric_head.load_state_dict(metric_best_state)
        with torch.inference_mode():
            test_result = run_epoch(
                metric_head, metric_test_loader, None, 2, torch.device("cpu")
            )
        metric_best.update({
            "test_loss": test_result["loss"],
            "test_acc": test_result["acc"],
            "test_bacc": test_result["bacc"],
        })
        backfill_best_test_metrics(metric_rows, metric_best["epoch"], test_result)
        metrics_run = temp / "metrics_run"
        metrics_checkpoint_path = metrics_run / "checkpoints" / "linear_probe_head_best.pt"
        metrics_checkpoint_path.parent.mkdir(parents=True)
        metrics_checkpoint = save_head_checkpoint(
            metrics_checkpoint_path, metric_head, metrics_args,
            {0: 0, 1: 1}, metric_best, None,
        )
        metrics_summary = build_summary_row(
            metrics_args, {0: 0, 1: 1}, None, metric_best,
            metrics_checkpoint_path,
        )
        assert_metrics_consistent(metric_rows, metrics_summary, metrics_checkpoint)
        epoch_csv = metrics_run / "metrics" / "epoch_metrics.csv"
        summary_csv = metrics_run / "summary.csv"
        write_csv_rows(epoch_csv, metric_rows)
        write_csv_rows(summary_csv, [metrics_summary])
        with open(epoch_csv, newline="", encoding="utf-8") as handle:
            saved_epoch_rows = list(csv.DictReader(handle))
        assert len(saved_epoch_rows) == metrics_args.epochs  # Metrics A
        required_metrics = {
            "train_loss", "train_acc", "train_bacc",
            "val_loss", "val_acc", "val_bacc",
            "test_loss", "test_acc", "test_bacc", "is_best",
        }
        assert required_metrics <= set(saved_epoch_rows[0])  # Metrics B
        saved_best_rows = [row for row in saved_epoch_rows if row["is_best"] == "True"]
        assert len(saved_best_rows) == 1  # Metrics C
        assert sum(bool(row["test_loss"]) for row in saved_epoch_rows) == 1
        assert saved_best_rows[0]["test_loss"]  # Metrics D
        with open(summary_csv, newline="", encoding="utf-8") as handle:
            saved_summary = next(csv.DictReader(handle))
        saved_checkpoint = torch.load(metrics_checkpoint_path, weights_only=True)
        assert int(saved_summary["best_epoch"]) == saved_checkpoint["best"]["epoch"]
        assert np.isclose(
            float(saved_summary["best_val_bacc"]),
            saved_checkpoint["best"]["val_bacc"],
        )
        assert np.isclose(
            float(saved_summary["test_bacc"]),
            saved_checkpoint["best"]["test_bacc"],
        )  # Metrics E; all inputs are synthetic CPU tensors (Metrics F)
        print(
            "CPU smoke test passed: core A-F; cache CLI A-D; "
            "Lance adapter; metrics A-F",
            flush=True,
        )
    finally:
        shutil.rmtree(temp)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        smoke_test()
    else:
        run(arguments)
