from functools import partial

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def fsdp_auto_wrap_policy():
    """Wrap each real Transformer encoder block as one FSDP unit."""
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={nn.TransformerEncoderLayer},
    )


def fsdp_mixed_precision_policy():
    """Production BF16 policy; CPU smoke tests intentionally do not use this."""
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )


def build_fsdp_model(model, device):
    """Build the CUDA production FSDP FULL_SHARD model wrapper."""
    return FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=fsdp_mixed_precision_policy(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
        device_id=device,
    )


def is_fsdp_model(model):
    return isinstance(model, FSDP)


def save_model_checkpoint(model, path, rank, extra_state):
    """
    Save a model-only checkpoint once on rank 0.

    All FSDP ranks must enter this function because full model-state gathering
    is collective. Optimizer state is deliberately neither gathered nor saved.
    Only rank 0 writes the resulting file.
    """
    if is_fsdp_model(model):
        state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=state_dict_config,
        ):
            model_state_dict = model.state_dict()
    else:
        model_state_dict = model.state_dict()

    if rank == 0:
        checkpoint = dict(extra_state)
        checkpoint["model_state_dict"] = model_state_dict
        checkpoint["checkpoint_version"] = 3
        checkpoint["checkpoint_format"] = "model_only"
        assert "optimizer_state_dict" not in checkpoint
        assert "scheduler_state_dict" not in checkpoint
        torch.save(checkpoint, path)
        print("[Checkpoint] format: model_only")
        print("[Checkpoint] model state: saved")
        print("[Checkpoint] optimizer state: not saved")
        print(f"[Checkpoint] epoch: {checkpoint.get('epoch')}")
        print(f"[Checkpoint] global_step: {checkpoint.get('global_step')}")
        print(f"Saved model checkpoint to {path}")


def load_model_state_dict(model, model_state_dict):
    """Load a full model state dict into a raw model or an FSDP model."""
    if is_fsdp_model(model):
        # Every FSDP rank must call this with the same full state dict.
        state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False,
        )
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=state_dict_config,
        ):
            return model.load_state_dict(model_state_dict)
    return model.load_state_dict(model_state_dict)


def fresh_training_state():
    """Return the epoch-level state for a run that does not resume."""
    return {
        "start_epoch": 0,
        "global_step": 0,
        "best_val": float("inf"),
        "optimizer_restored": False,
        "warnings": [],
    }


def restore_training_checkpoint(
    model,
    optimizer,
    path,
    steps_per_epoch,
):
    """
    Restore an FSDP-compatible checkpoint at an epoch boundary.

    New model-only checkpoints initialize a fresh optimizer, while legacy full
    checkpoints restore their optimizer state when it is present. Every FSDP
    rank must call this function and be able to read ``path``.
    """
    # Checkpoints include trusted training metadata such as the Config dataclass,
    # so PyTorch 2.6's default weights_only=True cannot deserialize them.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        raise ValueError(f"Checkpoint {path!r} does not contain model_state_dict.")
    load_model_state_dict(model, model_state_dict)

    warnings = []
    checkpoint_epoch = checkpoint.get("epoch")
    if isinstance(checkpoint_epoch, int) and checkpoint_epoch >= 0:
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0
        warnings.append(
            "checkpoint epoch is missing or not a completed epoch; starting at epoch 0"
        )

    optimizer_state_dict = checkpoint.get("optimizer_state_dict")
    optimizer_restored = optimizer_state_dict is not None
    if optimizer_restored:
        if is_fsdp_model(model):
            optimizer_state_dict = FSDP.optim_state_dict_to_load(
                optimizer_state_dict,
                model,
                optimizer,
            )
        optimizer.load_state_dict(optimizer_state_dict)
    else:
        warnings.append("optimizer state is not found; using freshly initialized AdamW")

    if "global_step" in checkpoint:
        global_step = int(checkpoint["global_step"])
    elif isinstance(checkpoint_epoch, int) and checkpoint_epoch >= 0:
        global_step = (checkpoint_epoch + 1) * int(steps_per_epoch)
        warnings.append(
            "global_step is not found; inferred it from completed epochs and steps_per_epoch"
        )
    else:
        global_step = 0
        warnings.append("global_step is not found; using 0")

    return {
        "checkpoint": checkpoint,
        "checkpoint_version": checkpoint.get("checkpoint_version", 1),
        "checkpoint_epoch": checkpoint_epoch,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "best_val": float(checkpoint.get("best_val", float("inf"))),
        "optimizer_restored": optimizer_restored,
        "warnings": warnings,
    }
