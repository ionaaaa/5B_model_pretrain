import numpy as np
import torch


def validate_mask_config(mask_config):
    """Validate the batch-level dynamic mask-ratio configuration."""
    mask_ratios = mask_config.mask_ratios
    mask_ratio_weights = mask_config.mask_ratio_weights

    if len(mask_ratios) == 0:
        raise ValueError("mask_ratios must contain at least one value.")
    if len(mask_ratios) != len(mask_ratio_weights):
        raise ValueError(
            "mask_ratios and mask_ratio_weights must have the same length."
        )
    if any(not 0.0 <= ratio < 1.0 for ratio in mask_ratios):
        raise ValueError("Each mask ratio must satisfy 0 <= mask_ratio < 1.")
    if any(weight <= 0.0 for weight in mask_ratio_weights):
        raise ValueError("Each mask ratio weight must be greater than 0.")


def sample_mask_ratio(mask_config, device=None) -> float:
    """Sample one mask ratio for one batch using PyTorch's seeded RNG."""
    validate_mask_config(mask_config)

    weights = torch.tensor(
        mask_config.mask_ratio_weights,
        dtype=torch.float32,
        device=device,
    )
    probabilities = weights / weights.sum()
    index = torch.multinomial(probabilities, num_samples=1, replacement=True)
    # Return the configured Python value exactly, while torch controls sampling.
    return float(mask_config.mask_ratios[index.item()])


def expected_mask_ratio(mask_config) -> float:
    """Return the weighted expected mask ratio for startup logging."""
    validate_mask_config(mask_config)
    total_weight = sum(mask_config.mask_ratio_weights)
    return sum(
        ratio * weight
        for ratio, weight in zip(mask_config.mask_ratios, mask_config.mask_ratio_weights)
    ) / total_weight


def generate_block_mask(
    num_tokens: int,
    mask_ratio: float,
):
    """
    生成随机起始位置的连续块 token mask。

    Returns:
        mask: [num_tokens]

    其中:
        1 表示该 token 被 mask
        0 表示该 token 保留
    """
    if num_tokens <= 0:
        return np.zeros((0,), dtype=np.float32)

    target_masked = int(round(num_tokens * mask_ratio))
    target_masked = max(0, min(num_tokens, target_masked))

    mask = np.zeros(num_tokens, dtype=np.float32)

    if target_masked == 0:
        return mask

    # 随机选择起始位置，确保 mask 块不超出边界
    max_start = num_tokens - target_masked
    mask_start = np.random.randint(0, max_start + 1)
    mask_end = mask_start + target_masked

    mask[mask_start:mask_end] = 1.0

    return mask


def apply_token_mask(tokens: torch.Tensor, token_mask: torch.Tensor):
    """
    在 token 维度上 mask。

    Args:
        tokens: [B, S, L]
        token_mask: [B, S]

    Returns:
        masked_tokens: [B, S, L]
    """
    masked_tokens = tokens.clone()

    masked_tokens[token_mask.bool()] = 0.0

    return masked_tokens
