from dataclasses import dataclass, field


@dataclass
class DataConfig:

    use_multi_lance_parts: bool = True

    set_paths: list = field(default_factory=lambda: [
        "data/sub01_01_EC.set",
    ])

    lance_path: str = "data/pretrain_processed.lance"

    lance_parts_dir: str = ""
    lance_part_list_file: str = "./part_paths.txt"
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
    mask_ratio: float = 0.75
    min_block_tokens: int = 4
    max_block_tokens: int = 32


@dataclass
class ModelConfig:
    """
    模型相关配置。
    """
    d_model: int = 2048
    n_heads: int = 16
    depth: int = 20
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
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    locality_window_batches: int = 16
    shuffle_within_window: bool = True

    use_prefetch_loader: bool = True
    prefetch: int = 2

    num_epochs: int = 5

    lr: float = 1e-4
    weight_decay: float = 1e-5

    print_every: int = 5

    output_root: str = "/vePFS-0x0d/home/mwx/workspace/EEG_pretrain/output"
    save_path: str = ""
    vis_dir: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
