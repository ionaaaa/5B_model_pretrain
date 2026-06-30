import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from configs import Config
from channel_config import STANDARD_64_CHANNELS

from dataset import collate_fn
from lance_dataset import LanceEEGPretrainDataset
from prefetch_loader import PrefetchLoader

from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from masking import generate_block_mask, apply_token_mask
from utils import set_seed, ensure_dir_for_file

from lance_dataset import (
    MultiLanceEEGPretrainDataset,
    PartAwareDistributedBatchSampler,
    load_storage_options,
    load_lance_part_paths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    args, _ = parser.parse_known_args()
    return args


def setup_distributed(local_rank_arg=None):
    """
    支持 torch.distributed.launch / torchrun 注入的环境变量。
    非分布式启动时自动退化为单进程模式。
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return False, 0, 1, 0, device

    local_rank = local_rank_arg
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def rank0_print(rank, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs)


def configure_run_outputs(cfg, rank, ddp):
    output_root = getattr(
        cfg.train,
        "output_root",
        "/vePFS-0x0d/home/mwx/workspace/EEG_pretrain/output",
    )

    run_name = None
    if is_main_process(rank):
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if ddp:
        obj_list = [run_name]
        dist.broadcast_object_list(obj_list, src=0)
        run_name = obj_list[0]

    run_dir = os.path.join(output_root, run_name)
    cfg.train.run_dir = run_dir
    cfg.train.save_path = os.path.join(run_dir, "checkpoints", "pretrain_checkpoint.pt")
    cfg.train.vis_dir = os.path.join(run_dir, "reconstruction_train")

    if is_main_process(rank):
        os.makedirs(os.path.dirname(cfg.train.save_path), exist_ok=True)
        os.makedirs(cfg.train.vis_dir, exist_ok=True)

    if ddp:
        dist.barrier()

    return run_dir


class BatchedSubset(Dataset):
    """
    保留底层 dataset.__getitems__ 批量读取能力的子集包装器。
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[int(idx)]]

    def __getitems__(self, idxs):
        mapped_indices = [self.indices[int(i)] for i in idxs]
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(mapped_indices)
        return [self.dataset[i] for i in mapped_indices]


def compute_shape_params(cfg):
    patch_len = int(round(cfg.data.patch_seconds * cfg.data.target_sfreq))
    patch_stride = int(round(cfg.data.patch_stride_seconds * cfg.data.target_sfreq))
    total_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))

    num_time_patches = int(
        np.floor((total_points - patch_len) / patch_stride) + 1
    )

    if num_time_patches <= 0:
        raise ValueError(
            f"num_time_patches <= 0. "
            f"Please check clip_seconds={cfg.data.clip_seconds}, "
            f"patch_seconds={cfg.data.patch_seconds}, "
            f"target_sfreq={cfg.data.target_sfreq}"
        )

    num_tokens = cfg.data.n_channels * num_time_patches
    frames_per_patch = int(round(cfg.data.patch_seconds / cfg.data.target_frame_seconds))
    n_bands = len(cfg.data.band_defs)

    return {
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "total_points": total_points,
        "num_time_patches": num_time_patches,
        "num_tokens": num_tokens,
        "frames_per_patch": frames_per_patch,
        "n_bands": n_bands,
    }


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = []
    for _ in range(batch_size):
        mask = generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
        )
        masks.append(mask)

    return torch.tensor(
        np.stack(masks, axis=0),
        dtype=torch.float32,
        device=device,
    )


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def visualize_train_reconstruction(
    model,
    loader,
    cfg,
    device,
    save_dir,
    max_samples=4,
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    token_inputs = batch["token_inputs"]
    targets = batch["targets"]
    token_channel_indices = batch["token_channel_indices"]
    token_time_indices = batch["token_time_indices"]
    token_valid_mask = batch["token_valid_mask"]

    B, S, _ = token_inputs.shape

    token_mask = build_batch_token_mask(B, S, cfg, device)
    masked_inputs = apply_token_mask(tokens=token_inputs, token_mask=token_mask)

    pred_norm = model(
        token_inputs=masked_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
    )

    pred = pred_norm.detach().cpu().numpy()
    target = targets.detach().cpu().numpy()
    token_mask_np = token_mask.detach().cpu().numpy()
    token_valid_np = token_valid_mask.detach().cpu().numpy()
    token_ch_np = token_channel_indices.detach().cpu().numpy()
    token_t_np = token_time_indices.detach().cpu().numpy()

    band_names = [x[0] for x in cfg.data.band_defs]
    num_to_plot = min(max_samples, B)

    for b in range(num_to_plot):
        valid_indices = np.where(token_valid_np[b] > 0.5)[0]
        masked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] > 0.5)
        )[0]
        unmasked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] < 0.5)
        )[0]

        if len(valid_indices) == 0:
            print(f"[Visualization] sample {b}: no valid token, skip.")
            continue
        if len(masked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no masked valid token, skip.")
            continue
        if len(unmasked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no unmasked valid token, skip.")
            continue

        masked_token_idx = int(np.random.choice(masked_valid_indices))
        unmasked_token_idx = int(np.random.choice(unmasked_valid_indices))

        examples = [("masked", masked_token_idx), ("unmasked", unmasked_token_idx)]

        F = target.shape[2]
        K = target.shape[3]
        x = np.arange(K)

        fig, axes = plt.subplots(
            nrows=F,
            ncols=2,
            figsize=(12, 2.2 * F),
            squeeze=False,
        )

        for col, (case_name, tok_idx) in enumerate(examples):
            ch_idx = int(token_ch_np[b, tok_idx])
            time_idx = int(token_t_np[b, tok_idx])
            ch_name = STANDARD_64_CHANNELS[ch_idx]

            for f in range(F):
                ax = axes[f][col]
                ax.plot(x, target[b, tok_idx, f], marker="o", linewidth=2, label="target")
                ax.plot(x, pred[b, tok_idx, f], marker="x", linewidth=2, label="pred")
                ax.set_title(
                    f"{case_name} | token={tok_idx} | {ch_name} | patch={time_idx} | {band_names[f]}"
                )
                ax.set_xlabel("frame inside patch")
                ax.set_ylabel("log band power")
                ax.grid(True, alpha=0.3)
                if f == 0:
                    ax.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"train_reconstruction_sample_{b}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Visualization] saved: {save_path}")


def build_dataset(cfg, rank):
    if cfg.data.use_multi_lance_parts:
        storage_options = load_storage_options(cfg.data.storage_options_path)
        part_paths = load_lance_part_paths(cfg.data.lance_part_list_file)
        if len(part_paths) == 0:
            raise ValueError(
                f"No Lance parts found in "
                f"{cfg.data.lance_part_list_file!r}."
            )
        rank0_print(rank, "lance_part_list_file:", cfg.data.lance_part_list_file)
        rank0_print(rank, "lance_parts:", len(part_paths))

        dataset = MultiLanceEEGPretrainDataset(
            lance_paths=part_paths,
            storage_options=storage_options,
        )
        rank0_print(rank, "num_lance_parts:", len(part_paths))
        return dataset

    rank0_print(rank, "lance_path:", cfg.data.lance_path)
    return LanceEEGPretrainDataset(
        lance_path=cfg.data.lance_path,
    )




def enable_cuda_attention_kernels(rank):
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    rank0_print(
        rank,
        "CUDA SDPA enabled: "
        f"flash={torch.backends.cuda.flash_sdp_enabled()}, "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
        f"math={torch.backends.cuda.math_sdp_enabled()}"
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    ddp, rank, world_size, local_rank, device = setup_distributed(args.local_rank)

    cfg = Config()
    run_dir = configure_run_outputs(cfg, rank, ddp)
    set_seed(cfg.train.seed + rank)
    enable_cuda_attention_kernels(rank)

    rank0_print(rank, "Using device:", device)
    rank0_print(rank, "Run output dir:", run_dir)
    rank0_print(rank, "Checkpoint path template:", cfg.train.save_path)
    rank0_print(rank, "Visualization dir:", cfg.train.vis_dir)
    rank0_print(
        rank,
        f"Distributed: {ddp}, rank={rank}, world_size={world_size}, local_rank={local_rank}",
    )

    shape_params = compute_shape_params(cfg)

    patch_len = shape_params["patch_len"]
    num_time_patches = shape_params["num_time_patches"]
    frames_per_patch = shape_params["frames_per_patch"]
    n_bands = shape_params["n_bands"]

    rank0_print(rank, "\n===== Shape Params =====")
    for k, v in shape_params.items():
        rank0_print(rank, f"{k}: {v}")

    rank0_print(rank, "\n===== Loading Processed EEG Samples From Lance =====")
    dataset = build_dataset(cfg, rank)

    rank0_print(rank, "dataset size:", len(dataset))

    if len(dataset) < 2:
        raise ValueError(
            "Too few samples. Need at least 2 for train/val split. "
            "Please regenerate Lance with shorter clip_seconds or use longer recordings."
        )

    if is_main_process(rank):
        s0 = dataset[0]

        print("\nFirst Lance sample:")
        print("  token_inputs:          ", tuple(s0["token_inputs"].shape))
        print("  targets:               ", tuple(s0["targets"].shape))
        print("  token_channel_indices: ", tuple(s0["token_channel_indices"].shape))
        print("  token_time_indices:    ", tuple(s0["token_time_indices"].shape))
        print("  token_valid_mask:      ", tuple(s0["token_valid_mask"].shape))
        print("  channel_valid_mask:    ", tuple(s0["channel_valid_mask"].shape))
        print("  valid channels:        ", int(s0["channel_valid_mask"].sum().item()), "/ 64")

    n_total = len(dataset)
    n_train = max(1, int(n_total * 0.8))
    n_val = n_total - n_train

    if n_val == 0:
        n_train = n_total - 1
        n_val = 1

    split_generator = torch.Generator().manual_seed(cfg.train.seed)
    all_indices = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    train_set = BatchedSubset(dataset, train_indices)
    val_set = BatchedSubset(dataset, val_indices)

    rank0_print(rank, f"\n===== Dataset Split =====")
    rank0_print(rank, f"train: {len(train_set)}, val: {len(val_set)}")

    train_batch_size = min(cfg.train.batch_size, len(train_set))
    val_batch_size = min(cfg.train.batch_size, len(val_set))

    rank0_print(rank, f"per-process train batch size: {train_batch_size}")
    rank0_print(rank, f"effective global train batch size: {train_batch_size * world_size}")

    train_batch_sampler = PartAwareDistributedBatchSampler(
        dataset=dataset,
        subset_indices=train_indices,
        batch_size=train_batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
        seed=cfg.train.seed,
        locality_window_batches=int(getattr(cfg.train, "locality_window_batches", 16)),
        shuffle_within_window=bool(getattr(cfg.train, "shuffle_within_window", True)),
    )

    val_batch_sampler = PartAwareDistributedBatchSampler(
        dataset=dataset,
        subset_indices=val_indices,
        batch_size=val_batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
        seed=cfg.train.seed,
        locality_window_batches=32,
        shuffle_within_window=False,
    )

    num_workers = int(getattr(cfg.train, "num_workers", 0))
    pin_memory = bool(getattr(cfg.train, "pin_memory", device.type == "cuda"))
    persistent_workers = bool(getattr(cfg.train, "persistent_workers", True))
    prefetch_factor = int(getattr(cfg.train, "prefetch_factor", 2))

    train_loader_kwargs = dict(
        dataset=train_set,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader_kwargs = dict(
        dataset=val_set,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    if num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
        val_loader_kwargs.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
        try:
            if "in_order" in DataLoader.__init__.__code__.co_varnames:
                train_loader_kwargs["in_order"] = False
                val_loader_kwargs["in_order"] = False
        except Exception:
            pass

    base_train_loader = DataLoader(**train_loader_kwargs)
    base_val_loader = DataLoader(**val_loader_kwargs)

    rank0_print(rank, f"PartAwareDistributedBatchSampler enabled: train_batches={len(train_batch_sampler)}, val_batches={len(val_batch_sampler)}")
    rank0_print(rank, f"DataLoader workers={num_workers}, pin_memory={pin_memory}, prefetch_factor={prefetch_factor if num_workers > 0 else None}")

    use_prefetch_loader = getattr(cfg.train, "use_prefetch_loader", True)
    prefetch = getattr(cfg.train, "prefetch", 2)

    if use_prefetch_loader:
        train_loader = PrefetchLoader(
            base_train_loader,
            device=device,
            prefetch=prefetch,
        )
        val_loader = PrefetchLoader(
            base_val_loader,
            device=device,
            prefetch=prefetch,
        )
        rank0_print(rank, f"PrefetchLoader enabled: prefetch={prefetch}")
    else:
        train_loader = base_train_loader
        val_loader = base_val_loader
        rank0_print(rank, "PrefetchLoader disabled")

    rank0_print(rank, "\n===== Building Model =====")
    model = EEGPretrainModel(
        input_dim=patch_len,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        depth=cfg.model.depth,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        n_channels=cfg.data.n_channels,
        n_time_patches=num_time_patches,
        n_bands=n_bands,
        frames_per_patch=frames_per_patch,
    ).to(device)

    total_params = count_parameters(model)

    rank0_print(rank, " ")
    rank0_print(rank, "--------------------------------------------------------------------------")
    rank0_print(rank, f"Total trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    rank0_print(rank, "--------------------------------------------------------------------------")
    rank0_print(rank, " ")

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTokenTrajectoryLoss()

    rank0_print(rank, "\n===== Start Training =====")
    best_val = float("inf")

    if is_main_process(rank):
        ensure_dir_for_file(cfg.train.save_path)
    if ddp:
        dist.barrier()

    for epoch in range(cfg.train.num_epochs):
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)

        rank0_print(rank, f"\n===== Epoch {epoch + 1}/{cfg.train.num_epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        rank0_print(rank, f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        base_dir = os.path.dirname(cfg.train.save_path)
        base_name = os.path.basename(cfg.train.save_path).replace(".pt", "")
        save_path = os.path.join(base_dir, f"{base_name}_{epoch}.pt")

        if is_main_process(rank):
            model_to_save = model.module if ddp else model
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "config": cfg,
                    "shape_params": shape_params,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                save_path,
            )

            print(f"Saved model to {save_path}")

        if val_loss < best_val:
            best_val = val_loss

    if is_main_process(rank):
        print("\n===== Visualizing Train Reconstruction =====")
        model_to_vis = model.module if ddp else model
        visualize_train_reconstruction(
            model=model_to_vis,
            loader=train_loader,
            cfg=cfg,
            device=device,
            save_dir=cfg.train.vis_dir,
            max_samples=4,
        )

        print("\nDone.")
        print(f"Best val loss: {best_val:.6f}")
        print(f"Visualization saved to: {cfg.train.vis_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
