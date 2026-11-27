from dataclasses import dataclass, field


@dataclass
class DataConfig:

    # ``processed_lance`` reads the historical FBMAE rows that already contain
    # tokens/targets. ``canonical_runtime`` reads canonical EEG and constructs
    # the same contract inside DataLoader workers.
    input_mode: str = "processed_lance"
    canonical_dataset_key: str = "egobrain"
    canonical_lance_path: str = ""
    runtime_max_samples: int | None = None

    use_multi_lance_parts: bool = True

    set_paths: list = field(default_factory=lambda: [
        "data/sub01_01_EC.set",
    ])

    lance_path: str = "data/pretrain_processed.lance"

    lance_parts_dir: str = ""
    lance_part_list_file: str = "./part_paths.txt"
    processed_preflight_rows_per_part: int = 3
    processed_preflight_wait_timeout_seconds: int = 7200
    storage_options_path: str = "./storage_options.json"

    convert_v_to_uv: bool = True
    clip_seconds: float = 10.0
    clip_stride_seconds: float = 10.0
    target_sfreq: float = 100.0
    n_channels: int = 64
    patch_seconds: float = 1.0
    patch_stride_seconds: float = 1.0
    target_frame_seconds: float = 0.2
    stft_window_seconds: float = 1.0
    stft_hop_seconds: float = 0.1
    band_defs: list = field(default_factory=lambda: [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma_low", 30.0, 45.0),
    ])
    eps: float = 1e-6






@dataclass
class MaskConfig:
    """
    mask 相关配置。
    """
    # Formal single-dataset runtime pretraining uses a fixed 80% mask ratio.
    mask_ratios: list[float] = field(default_factory=lambda: [0.8])
    mask_ratio_weights: list[float] = field(default_factory=lambda: [1.0])
    min_block_tokens: int = 4
    max_block_tokens: int = 32


@dataclass
class ModelConfig:
    """
    模型相关配置。
    """
    profile: str = "50m"
    d_model: int = 512
    n_heads: int = 8
    depth: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """
    训练相关配置。
    """
    seed: int = 42
    device: str = "cuda"

    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    locality_window_batches: int = 16
    shuffle_within_window: bool = True

    use_prefetch_loader: bool = True
    prefetch: int = 2

    num_epochs: int = 1

    lr: float = 1e-4
    weight_decay: float = 1e-5

    print_every: int = 5
    # Per-rank, non-synchronizing diagnostics for the final batches of an epoch.
    # Set to 0 to disable StepDebug logging.
    step_debug_last_n: int = 50

    output_root: str = "/tos/mnt/LargeScalePretrainResults_5B"
    save_path: str = "/tos/mnt/LargeScalePretrainResults_5B"
    vis_dir: str = "/tos/mnt/LargeScalePretrainResults_5B"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# Profiles keep the historical 5B settings available without making them the
# accidental default for runtime-adaptation smoke/pretraining runs.
PRETRAIN_PROFILES = {
    "50m": {
        "model": {
            "d_model": 512,
            "n_heads": 8,
            "depth": 12,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 64,
            "num_workers": 0,
            "num_epochs": 1,
            "lr": 1e-4,
            "weight_decay": 1e-5,
            "print_every": 5,
        },
    },
    "5b": {
        "model": {
            "d_model": 4096,
            "n_heads": 16,
            "depth": 25,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 32,
            "num_workers": 8,
            "num_epochs": 10,
            "lr": 5e-5,
            "weight_decay": 1e-5,
            "print_every": 50,
        },
    },
}


def apply_pretrain_profile(cfg: Config, profile_name: str) -> Config:
    """Apply one named model/training profile to an existing config."""
    if profile_name not in PRETRAIN_PROFILES:
        raise ValueError(
            f"Unknown pretraining profile {profile_name!r}; "
            f"available={sorted(PRETRAIN_PROFILES)}"
        )
    profile = PRETRAIN_PROFILES[profile_name]
    cfg.model.profile = profile_name
    for name, value in profile["model"].items():
        setattr(cfg.model, name, value)
    for name, value in profile["train"].items():
        setattr(cfg.train, name, value)
    return cfg
