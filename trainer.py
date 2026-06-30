import torch
from contextlib import nullcontext
import torch.distributed as dist
import numpy as np
from masking import generate_block_mask, apply_token_mask


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return (not is_distributed()) or dist.get_rank() == 0


def autocast_context(device):
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def reduce_mean(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor.item()


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = np.stack([
        generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
        )
        for _ in range(batch_size)
    ], axis=0)

    return torch.from_numpy(masks).float().to(device)


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


def forward_loss(
    model,
    criterion,
    masked_inputs,
    targets,
    token_channel_indices,
    token_time_indices,
    token_valid_mask,
    device,
):
    with autocast_context(device):
        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        loss = criterion(
            pred=pred,
            target=targets,
            token_valid_mask=token_valid_mask,
        )
    return loss


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg,
):
    model.train()
    total_loss = 0.0
    total_steps = len(loader)

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]                    # [B, S, L]
        targets = batch["targets"]                              # [B, S, F, K]
        token_channel_indices = batch["token_channel_indices"]  # [B, S]
        token_time_indices = batch["token_time_indices"]        # [B, S]
        token_valid_mask = batch["token_valid_mask"]            # [B, S]

        B, S, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=B,
            num_tokens=S,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        loss = forward_loss(
            model=model,
            criterion=criterion,
            masked_inputs=masked_inputs,
            targets=targets,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            token_valid_mask=token_valid_mask,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        current_step = step + 1
        should_print = (
            current_step == 1
            or current_step % cfg.train.print_every == 0
            or current_step == total_steps
        )
        if is_main_process() and should_print:
            progress = 100.0 * current_step / max(1, total_steps)
            print(
                f"step={current_step}/{total_steps} "
                f"({progress:.2f}%), "
                f"loss={loss.item():.6f}, "
                f"input_mask_ratio={token_mask.mean().item():.3f}"
            )

    local_avg = total_loss / max(1, len(loader))
    return reduce_mean(local_avg, device)


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    cfg,
):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]
        targets = batch["targets"]
        token_channel_indices = batch["token_channel_indices"]
        token_time_indices = batch["token_time_indices"]
        token_valid_mask = batch["token_valid_mask"]

        B, S, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=B,
            num_tokens=S,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        loss = forward_loss(
            model=model,
            criterion=criterion,
            masked_inputs=masked_inputs,
            targets=targets,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
            token_valid_mask=token_valid_mask,
            device=device,
        )

        total_loss += loss.item()

    local_avg = total_loss / max(1, len(loader))
    return reduce_mean(local_avg, device)
