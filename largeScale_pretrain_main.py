import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from configs import Config, PRETRAIN_PROFILES, apply_pretrain_profile
from channel_config import STANDARD_64_CHANNELS

from dataset import collate_fn
from canonical_runtime_dataset import CanonicalRuntimeTargetDataset
from lance_dataset import LanceEEGPretrainDataset
from prefetch_loader import PrefetchLoader

from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from fsdp_utils import (
    build_fsdp_model,
    fresh_training_state,
    restore_training_checkpoint,
    save_model_checkpoint,
)
from trainer import train_one_epoch, validate_one_epoch
from masking import (
    generate_block_mask,
    apply_token_mask,
    expected_mask_ratio,
    sample_mask_ratio,
    validate_mask_config,
)
from utils import set_seed, ensure_dir_for_file

from lance_dataset import (
    MultiLanceEEGPretrainDataset,
    PartAwareDistributedBatchSampler,
    is_remote_lance_path,
    load_storage_options,
    load_lance_part_paths,
    preflight_processed_lance_parts,
)


NCCL_FLIGHT_RECORDER_DEFAULTS = {
    "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
    "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
    "TORCH_NCCL_DESYNC_DEBUG": "1",
    "TORCH_NCCL_TRACE_CPP_STACK": "1",
}


def configure_nccl_flight_recorder_defaults():
    """Set diagnostic defaults before ProcessGroupNCCL initialization.

    ``setdefault`` deliberately preserves values supplied by the launch shell.
    """
    for name, value in NCCL_FLIGHT_RECORDER_DEFAULTS.items():
        os.environ.setdefault(name, value)


def log_nccl_flight_recorder_status(rank):
    """Report the actual effective diagnostic environment on rank 0."""
    if not is_main_process(rank):
        return
    print("[NCCL Debug]")
    print(f"PyTorch={torch.__version__}")
    for name in NCCL_FLIGHT_RECORDER_DEFAULTS:
        print(f"{name}={os.environ.get(name, '<unset>')}")


def resolve_use_prefetch_loader(cli_value, config_value):
    """Use an explicit CLI selection when present, otherwise keep config behavior."""
    return bool(config_value) if cli_value is None else bool(cli_value)


def maybe_wrap_prefetch_loaders(
    train_loader,
    val_loader,
    *,
    use_prefetch_loader,
    device,
    prefetch,
):
    """Optionally add only the project's custom PrefetchLoader wrapper."""
    if not use_prefetch_loader:
        return train_loader, val_loader
    return (
        PrefetchLoader(train_loader, device=device, prefetch=prefetch),
        PrefetchLoader(val_loader, device=device, prefetch=prefetch),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", "--local-rank", type=int, default=None)
    parser.add_argument(
        "--model-profile",
        choices=tuple(PRETRAIN_PROFILES),
        default=None,
        help="Model/training scale profile. Defaults to the config profile (50m).",
    )
    parser.add_argument(
        "--data-mode",
        choices=("processed_lance", "canonical_runtime"),
        default=None,
        help="Read historical processed tokens or adapt canonical EEG at runtime.",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        default=None,
        help="Canonical dataset identity, e.g. inria, egobrain, or shin2017a.",
    )
    parser.add_argument(
        "--lance-path",
        type=str,
        default=None,
        help="Mounted single Lance path used by canonical_runtime mode.",
    )
    parser.add_argument(
        "--part-list-file",
        type=str,
        default=None,
        help="Text file containing one mounted processed Lance path per line.",
    )
    parser.add_argument(
        "--preflight-rows-per-part",
        type=int,
        default=None,
        help="Rows sampled per processed Lance during startup schema/shape/finite checks.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional number of full canonical rows for a bounded trial run.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an epoch-complete FSDP/full-model checkpoint.",
    )
    parser.add_argument(
        "--step-debug-last-n",
        type=int,
        default=None,
        help="Emit per-rank StepDebug markers for the final N training batches; 0 disables it.",
    )
    prefetch_group = parser.add_mutually_exclusive_group()
    prefetch_group.add_argument(
        "--use-prefetch-loader",
        dest="use_prefetch_loader",
        action="store_true",
        help="Force the project's custom PrefetchLoader on.",
    )
    prefetch_group.add_argument(
        "--no-prefetch-loader",
        dest="use_prefetch_loader",
        action="store_false",
        help="Disable only the project's custom PrefetchLoader wrapper.",
    )
    parser.set_defaults(use_prefetch_loader=None)
    args, _ = parser.parse_known_args()
    return args


def apply_cli_config(cfg, args):
    """Apply a named profile first, then explicit trial/run overrides."""
    profile_name = args.model_profile or cfg.model.profile
    apply_pretrain_profile(cfg, profile_name)

    if args.data_mode is not None:
        cfg.data.input_mode = args.data_mode
    if args.dataset_key is not None:
        cfg.data.canonical_dataset_key = args.dataset_key
    if args.lance_path is not None:
        cfg.data.canonical_lance_path = args.lance_path
    if args.part_list_file is not None:
        cfg.data.lance_part_list_file = args.part_list_file
        cfg.data.use_multi_lance_parts = True
    if args.preflight_rows_per_part is not None:
        if args.preflight_rows_per_part <= 0:
            raise ValueError("--preflight-rows-per-part must be positive")
        cfg.data.processed_preflight_rows_per_part = args.preflight_rows_per_part
    if args.max_samples is not None:
        if args.max_samples < 2:
            raise ValueError("--max-samples must be at least 2 for train/val split")
        cfg.data.runtime_max_samples = args.max_samples
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("--num-workers must be non-negative")
        cfg.train.num_workers = args.num_workers
    if args.num_epochs is not None:
        if args.num_epochs <= 0:
            raise ValueError("--num-epochs must be positive")
        cfg.train.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        cfg.train.lr = args.learning_rate
    if args.output_root is not None:
        cfg.train.output_root = args.output_root
    return cfg


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


def configure_run_outputs(cfg, rank, distributed):
    output_root = getattr(
        cfg.train,
        "output_root",
        "/vePFS-0x0d/home/mwx/workspace/EEG_pretrain/output",
    )

    run_name = None
    if is_main_process(rank):
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    if distributed:
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

    if distributed:
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


def build_batch_token_mask(batch_size, num_tokens, mask_ratio, device):
    masks = []
    for _ in range(batch_size):
        mask = generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=mask_ratio,
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
    save_outputs=True,
    artifact_tag=None,
):
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    token_inputs = batch["token_inputs"]
    targets = batch["targets"]
    token_channel_indices = batch["token_channel_indices"]
    token_time_indices = batch["token_time_indices"]
    token_valid_mask = batch["token_valid_mask"]

    B, S, _ = token_inputs.shape

    mask_ratio = sample_mask_ratio(cfg.mask, device=device)
    token_mask = build_batch_token_mask(B, S, mask_ratio, device)
    masked_inputs = apply_token_mask(tokens=token_inputs, token_mask=token_mask)

    pred_norm = model(
        token_inputs=masked_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
    )

    # FSDP requires every rank to participate in the forward above. Only rank 0
    # needs the following CPU conversion and image output.
    if not save_outputs:
        return

    os.makedirs(save_dir, exist_ok=True)

    # NumPy cannot directly represent CUDA/CPU bfloat16 tensors produced by
    # FSDP BF16 mixed precision; FP32 is only used for plotting.
    pred = pred_norm.detach().float().cpu().numpy()
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
        file_stem = "train_reconstruction_sample"
        if artifact_tag:
            file_stem = f"{artifact_tag}_{file_stem}"
        save_path = os.path.join(save_dir, f"{file_stem}_{b}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[Visualization] saved: {save_path}")


def build_dataset(cfg, rank):
    input_mode = getattr(cfg.data, "input_mode", "processed_lance")
    if input_mode == "canonical_runtime":
        lance_path = str(cfg.data.canonical_lance_path)
        dataset_key = str(cfg.data.canonical_dataset_key)
        if not lance_path:
            raise ValueError(
                "canonical_runtime requires --lance-path or "
                "cfg.data.canonical_lance_path"
            )
        if not os.path.exists(lance_path):
            raise FileNotFoundError(
                f"Mounted canonical Lance path does not exist: {lance_path}"
            )
        dataset = CanonicalRuntimeTargetDataset(
            lance_path=lance_path,
            dataset_key=dataset_key,
            max_samples=cfg.data.runtime_max_samples,
            validate=True,
        )
        rank0_print(rank, "[CanonicalRuntime] dataset_key:", dataset_key)
        rank0_print(rank, "[CanonicalRuntime] lance_path:", lance_path)
        rank0_print(rank, "[CanonicalRuntime] valid_samples_scan:", dataset.valid_samples_audit)
        rank0_print(rank, "[CanonicalRuntime] channel_mapping_audit:", dataset.mapping_audit)
        rank0_print(rank, "[CanonicalRuntime] targets: generated in DataLoader workers")
        return dataset

    if input_mode != "processed_lance":
        raise ValueError(f"Unsupported data input_mode={input_mode!r}")

    if cfg.data.use_multi_lance_parts:
        part_paths = load_lance_part_paths(cfg.data.lance_part_list_file)
        if len(part_paths) == 0:
            raise ValueError(
                f"No Lance parts found in "
                f"{cfg.data.lance_part_list_file!r}."
            )

        first_part_path = part_paths[0]
        has_remote_parts = any(is_remote_lance_path(path) for path in part_paths)
        storage_options = (
            load_storage_options(cfg.data.storage_options_path)
            if has_remote_parts
            else None
        )

        rank0_print(rank, "[MultiLance] lance_part_list_file:", cfg.data.lance_part_list_file)
        rank0_print(rank, f"[MultiLance] loaded {len(part_paths)} lance parts")
        rank0_print(rank, "[MultiLance] first part:", first_part_path)

        if not is_remote_lance_path(first_part_path) and not os.path.exists(first_part_path):
            raise FileNotFoundError(
                "Lance part does not exist:\n"
                f"  {first_part_path}\n\n"
                "Expected mount root:\n"
                "  /tos/mnt"
            )

        shape_params = compute_shape_params(cfg)
        rows_per_part = int(getattr(cfg.data, "processed_preflight_rows_per_part", 3))
        distributed = dist.is_available() and dist.is_initialized()
        payload = [None]
        if (not distributed) or rank == 0:
            try:
                payload[0] = {
                    "report": preflight_processed_lance_parts(
                        lance_paths=part_paths,
                        storage_options=storage_options,
                        sample_rows_per_part=rows_per_part,
                        expected_num_tokens=shape_params["num_tokens"],
                        expected_patch_len=shape_params["patch_len"],
                        expected_n_bands=shape_params["n_bands"],
                        expected_frames_per_patch=shape_params["frames_per_patch"],
                        expected_n_channels=cfg.data.n_channels,
                    )
                }
            except Exception as exc:
                payload[0] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        if payload[0].get("error"):
            raise RuntimeError(
                "Processed Lance startup preflight failed before model construction: "
                + payload[0]["error"]
            )

        preflight = payload[0]["report"]
        rank0_print(
            rank,
            "[ProcessedPreflight] PASS "
            f"parts={preflight['parts']} total_rows={preflight['total_rows']} "
            f"sampled_rows={preflight['sampled_rows']} "
            f"elapsed_s={preflight['elapsed_seconds']:.3f}",
        )
        rank0_print(rank, "[ProcessedPreflight] expected_shapes:", preflight["expected_shapes"])
        for part_report in preflight["part_reports"]:
            rank0_print(
                rank,
                "[ProcessedPreflight] "
                f"part={part_report['part_id']} rows={part_report['rows']} "
                f"sampled={part_report['sampled_indices']} "
                f"valid_channels={part_report['valid_channel_counts']} "
                f"path={part_report['path']}",
            )

        dataset = MultiLanceEEGPretrainDataset(
            lance_paths=part_paths,
            storage_options=storage_options,
            lengths=preflight["lengths"],
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
    # ProcessGroupNCCL reads its diagnostics configuration during initialization.
    # torchrun starts this Python process before that point, so this is early
    # enough while still preserving explicit launcher-provided values.
    configure_nccl_flight_recorder_defaults()
    distributed, rank, world_size, local_rank, device = setup_distributed(args.local_rank)

    cfg = apply_cli_config(Config(), args)
    if args.step_debug_last_n is not None:
        if args.step_debug_last_n < 0:
            raise ValueError("--step-debug-last-n must be greater than or equal to 0")
        cfg.train.step_debug_last_n = args.step_debug_last_n
    cfg.train.use_prefetch_loader = resolve_use_prefetch_loader(
        args.use_prefetch_loader,
        cfg.train.use_prefetch_loader,
    )
    validate_mask_config(cfg.mask)
    run_dir = configure_run_outputs(cfg, rank, distributed)
    set_seed(cfg.train.seed + rank)
    enable_cuda_attention_kernels(rank)

    rank0_print(rank, "Using device:", device)
    rank0_print(rank, "Model profile:", cfg.model.profile)
    rank0_print(rank, "Data input mode:", cfg.data.input_mode)
    rank0_print(rank, "Run output dir:", run_dir)
    rank0_print(rank, "Checkpoint path template:", cfg.train.save_path)
    rank0_print(rank, "Visualization dir:", cfg.train.vis_dir)
    rank0_print(
        rank,
        f"Distributed: {distributed}, rank={rank}, world_size={world_size}, local_rank={local_rank}",
    )
    if distributed:
        log_nccl_flight_recorder_status(rank)
    rank0_print(rank, f"StepDebug final batches per epoch: {cfg.train.step_debug_last_n}")

    shape_params = compute_shape_params(cfg)

    patch_len = shape_params["patch_len"]
    num_time_patches = shape_params["num_time_patches"]
    frames_per_patch = shape_params["frames_per_patch"]
    n_bands = shape_params["n_bands"]

    rank0_print(rank, "\n===== Shape Params =====")
    for k, v in shape_params.items():
        rank0_print(rank, f"{k}: {v}")

    rank0_print(rank, "\n===== Loading EEG Samples From Lance =====")
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
        if "channel_mapping_policy" in s0:
            print("  mapping policy:        ", s0["channel_mapping_policy"])
            print("  dropped channels:      ", s0["dropped_channel_names"] or "NONE")

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
    use_prefetch_loader = cfg.train.use_prefetch_loader
    prefetch = getattr(cfg.train, "prefetch", 2)
    train_loader, val_loader = maybe_wrap_prefetch_loaders(
        base_train_loader,
        base_val_loader,
        use_prefetch_loader=use_prefetch_loader,
        device=device,
        prefetch=prefetch,
    )

    rank0_print(
        rank,
        "[DataLoaderConfig] "
        f"use_prefetch_loader={use_prefetch_loader} "
        f"num_workers={num_workers} "
        f"prefetch_factor={prefetch_factor if num_workers > 0 else None} "
        f"persistent_workers={persistent_workers if num_workers > 0 else False} "
        f"pin_memory={pin_memory}",
    )
    if use_prefetch_loader:
        rank0_print(rank, f"[DataLoaderConfig] custom PrefetchLoader enabled; prefetch={prefetch}")
    else:
        rank0_print(
            rank,
            "[DataLoaderConfig] custom PrefetchLoader disabled; using raw PyTorch DataLoader",
        )

    rank0_print(rank, "\n===== Building Model =====")
    raw_model = EEGPretrainModel(
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
    )

    total_params = count_parameters(raw_model)

    rank0_print(rank, " ")
    rank0_print(rank, "--------------------------------------------------------------------------")
    rank0_print(rank, f"Total trainable parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    rank0_print(rank, "--------------------------------------------------------------------------")
    rank0_print(rank, " ")

    rank0_print(rank, f"Dynamic mask ratios: {cfg.mask.mask_ratios}")
    rank0_print(rank, f"Dynamic mask weights: {cfg.mask.mask_ratio_weights}")
    rank0_print(rank, f"Expected mask ratio: {expected_mask_ratio(cfg.mask):.4f}")
    rank0_print(rank, "Sampling frequency: once per training batch")

    if distributed:
        model = build_fsdp_model(raw_model, device=device)
        print(
            f"[FSDP] rank={rank} local_rank={local_rank} device={device}",
            flush=True,
        )
        rank0_print(rank, "Distributed mode: FSDP")
        rank0_print(rank, "FSDP sharding strategy: FULL_SHARD")
        rank0_print(rank, "FSDP auto-wrap block: TransformerEncoderLayer")
        rank0_print(rank, "FSDP mixed precision: BF16")
        rank0_print(rank, "Activation checkpointing: disabled")
        rank0_print(rank, "CPU offload: disabled")
        rank0_print(rank, f"World size: {world_size}")
    else:
        model = raw_model.to(device)
        rank0_print(rank, "Distributed mode: single-process (FSDP inactive)")

    # Optimizer construction deliberately follows FSDP wrapping.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTokenTrajectoryLoss()

    training_state = fresh_training_state()
    if args.resume:
        training_state = restore_training_checkpoint(
            model=model,
            optimizer=optimizer,
            path=args.resume,
            steps_per_epoch=len(train_loader),
        )
        rank0_print(rank, "\n===== Resume Training =====")
        rank0_print(rank, f"[Resume] checkpoint: {args.resume}")
        rank0_print(rank, f"[Resume] checkpoint version: {training_state['checkpoint_version']}")
        rank0_print(rank, f"[Resume] checkpoint epoch: {training_state['checkpoint_epoch']}")
        rank0_print(rank, f"[Resume] start epoch: {training_state['start_epoch']}")
        rank0_print(rank, "[Resume] model state: restored")
        if training_state["optimizer_restored"]:
            rank0_print(rank, "[Resume] optimizer state: restored")
        else:
            rank0_print(
                rank,
                "[Resume] optimizer state: not found; using freshly initialized AdamW",
            )
        rank0_print(rank, f"[Resume] global step: {training_state['global_step']}")
        for warning in training_state["warnings"]:
            rank0_print(rank, f"[Resume] warning: {warning}")

    start_epoch = training_state["start_epoch"]
    global_step = training_state["global_step"]
    best_val = training_state["best_val"]

    rank0_print(rank, "\n===== Start Training =====")

    if is_main_process(rank):
        ensure_dir_for_file(cfg.train.save_path)
    if distributed:
        dist.barrier()

    base_dir = os.path.dirname(cfg.train.save_path)
    base_name = os.path.basename(cfg.train.save_path).replace(".pt", "")

    if args.resume is None:
        # Save the randomly initialized model and its reconstruction before any
        # optimization step. FSDP checkpointing/forward remains collective.
        initial_checkpoint_path = os.path.join(base_dir, f"{base_name}_initial.pt")
        save_model_checkpoint(
            model=model,
            path=initial_checkpoint_path,
            rank=rank,
            extra_state={
                "config": cfg,
                "shape_params": shape_params,
                "epoch": None,
                "phase": "before_training",
                "global_step": global_step,
                "best_val": best_val,
                "train_loss": None,
                "val_loss": None,
            },
        )
        if is_main_process(rank):
            print(f"Saved initial model to {initial_checkpoint_path}")
        if distributed:
            dist.barrier()

        if is_main_process(rank):
            print("\n===== Visualizing Initial Reconstruction =====")
        visualize_train_reconstruction(
            model=model,
            loader=train_loader,
            cfg=cfg,
            device=device,
            save_dir=cfg.train.vis_dir,
            max_samples=4,
            save_outputs=is_main_process(rank),
            artifact_tag="before_training",
        )
        if distributed:
            dist.barrier()
    else:
        rank0_print(rank, "[Resume] skipping fresh-run initial checkpoint and visualization")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        if hasattr(train_batch_sampler, "set_epoch"):
            train_batch_sampler.set_epoch(epoch)

        rank0_print(rank, f"\n===== Epoch {epoch + 1}/{cfg.train.num_epochs} =====")

        train_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            global_step=global_step,
            epoch=epoch,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        rank0_print(rank, f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss

        save_path = os.path.join(base_dir, f"{base_name}_{epoch}.pt")

        save_model_checkpoint(
            model=model,
            path=save_path,
            rank=rank,
            extra_state={
                "config": cfg,
                "shape_params": shape_params,
                "epoch": epoch,
                "global_step": global_step,
                "best_val": best_val,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )
        if is_main_process(rank):
            print(f"Saved model to {save_path}")
        if distributed:
            dist.barrier()

        if is_main_process(rank):
            print(f"\n===== Visualizing Epoch {epoch + 1} Reconstruction =====")
        visualize_train_reconstruction(
            model=model,
            loader=train_loader,
            cfg=cfg,
            device=device,
            save_dir=cfg.train.vis_dir,
            max_samples=4,
            save_outputs=is_main_process(rank),
            artifact_tag=f"epoch_{epoch + 1:03d}",
        )
        if distributed:
            dist.barrier()

    if is_main_process(rank):
        print("\n===== Visualizing Train Reconstruction =====")
    # FSDP forward is collective, so all ranks participate; only rank 0 writes.
    visualize_train_reconstruction(
        model=model,
        loader=train_loader,
        cfg=cfg,
        device=device,
        save_dir=cfg.train.vis_dir,
        max_samples=4,
        save_outputs=is_main_process(rank),
        artifact_tag="final",
    )
    if distributed:
        dist.barrier()

    if is_main_process(rank):
        print("\nDone.")
        print(f"Best val loss: {best_val:.6f}")
        print(f"Visualization saved to: {cfg.train.vis_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
