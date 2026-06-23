from typing import Dict, Optional

import torch
import torch.nn.functional as F

# "Masked" signifies that the loss is averaged only over valid pixels, as indicated by the mask.

EPS = 1e-6

def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    mask = mask.float()
    return (x * mask).sum() / (mask.sum() + eps)

def masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    loss = torch.abs(pred - target)
    return masked_mean(loss, mask, eps=eps)

def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    loss = (pred - target) ** 2
    return masked_mean(loss, mask, eps=eps)

def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask, eps=eps) + eps)

def masked_bias(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    return masked_mean(pred - target, mask, eps=eps)

def laplacian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Optional structure loss at 32x32.
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=pred.dtype,
        device=pred.device,
    ).view(1, 1, 3, 3)

    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)

    if mask is None:
        return F.l1_loss(pred_lap, target_lap)

    return masked_l1(pred_lap, target_lap, mask, eps=eps)

def sd_weight_from_raw(
    sd: torch.Tensor,
    target_raw: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    power: float = 0.5,
    clip: float = 10.0,
    logspace: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Build finite uncertainty weights from ESA CCI SD.

    High SD means low confidence, hence lower weight.
    """

    valid = valid_mask > 0.5

    sd_safe = torch.where(valid, sd, torch.zeros_like(sd))
    target_safe = torch.where(valid, target_raw, torch.zeros_like(target_raw))

    sd_safe = torch.nan_to_num(sd_safe, nan=0.0, posinf=0.0, neginf=0.0)
    target_safe = torch.nan_to_num(target_safe, nan=0.0, posinf=0.0, neginf=0.0)

    sd_safe = sd_safe.clamp_min(0.0)
    target_safe = target_safe.clamp_min(0.0)

    if logspace:
        uncertainty = sd_safe / (target_safe + 1.0)
    else:
        uncertainty = sd_safe

    uncertainty = torch.nan_to_num(
        uncertainty,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    uncertainty = uncertainty.clamp_min(eps)

    weight = uncertainty.pow(-power)

    weight = torch.nan_to_num(
        weight,
        nan=0.0,
        posinf=clip,
        neginf=0.0,
    )

    weight = weight.clamp(max=clip)

    # Completely remove invalid pixels from the weighted loss.
    weight = torch.where(valid, weight, torch.zeros_like(weight))

    return weight


def weighted_masked_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    valid = mask > 0.5

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

    weight = torch.where(valid, weight, torch.zeros_like(weight))

    diff = torch.abs(pred - target)
    diff = torch.where(valid, diff, torch.zeros_like(diff))

    numerator = (diff * weight).sum()
    denominator = weight.sum().clamp_min(eps)

    return numerator / denominator


def weighted_masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    valid = mask > 0.5

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

    weight = torch.where(valid, weight, torch.zeros_like(weight))

    diff2 = (pred - target).pow(2)
    diff2 = torch.where(valid, diff2, torch.zeros_like(diff2))

    numerator = (diff2 * weight).sum()
    denominator = weight.sum().clamp_min(eps)

    return numerator / denominator


def sd_weighted_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sd: torch.Tensor,
    mask: torch.Tensor,
    power: float = 0.5,
    clip: float = 10.0,
    logspace: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    weight = sd_weight_from_raw(
        sd=sd,
        target_raw=target_raw,
        valid_mask=mask,
        power=power,
        clip=clip,
        logspace=logspace,
        eps=eps,
    )

    return weighted_masked_l1(
        pred=pred,
        target=target,
        mask=mask,
        weight=weight,
        eps=eps,
    )


def sd_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sd: torch.Tensor,
    mask: torch.Tensor,
    power: float = 0.5,
    clip: float = 10.0,
    logspace: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    weight = sd_weight_from_raw(
        sd=sd,
        target_raw=target_raw,
        valid_mask=mask,
        power=power,
        clip=clip,
        logspace=logspace,
        eps=eps,
    )

    return weighted_masked_mse(
        pred=pred,
        target=target,
        mask=mask,
        weight=weight,
        eps=eps,
    )


def make_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_raw: torch.Tensor,
    sd: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
    laplacian_coeff: float = 0.0,
    sd_weight_power: float = 0.5,
    sd_weight_clip: float = 10.0,
    sd_logspace: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Main loss used by train_eval.py.

    pred and target are usually normalized transformed AGB tensors.
    target_raw and sd remain in raw AGB units.
    """
    if loss_type == "l1":
        main = masked_l1(pred, target, mask)

    elif loss_type == "mse":
        main = masked_mse(pred, target, mask)

    elif loss_type == "sd_weighted_l1":
        main = sd_weighted_l1(
            pred=pred,
            target=target,
            target_raw=target_raw,
            sd=sd,
            mask=mask,
            power=sd_weight_power,
            clip=sd_weight_clip,
            logspace=sd_logspace,
        )

    elif loss_type == "sd_weighted_mse":
        main = sd_weighted_mse(
            pred=pred,
            target=target,
            target_raw=target_raw,
            sd=sd,
            mask=mask,
            power=sd_weight_power,
            clip=sd_weight_clip,
            logspace=sd_logspace,
        )

    lap = pred.new_tensor(0.0)
    if laplacian_coeff > 0:
        lap = laplacian_loss(pred, target, mask)

    total = main + float(laplacian_coeff) * lap

    return {
        "loss": total,
        "main": main.detach(),
        "laplacian": lap.detach(),
    }

@torch.no_grad()
def regression_metrics(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Metrics in raw AGB units, measured directly on the 32x32 ESA grid.
    """
    mae = masked_l1(pred_raw, target_raw, mask)
    rmse = masked_rmse(pred_raw, target_raw, mask)
    bias = masked_bias(pred_raw, target_raw, mask)

    valid = mask.float()
    pred_mean = masked_mean(pred_raw, valid)
    target_mean = masked_mean(target_raw, valid)

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "bias": float(bias.item()),
        "pred_mean": float(pred_mean.item()),
        "target_mean": float(target_mean.item()),
    }


@torch.no_grad()
def batch_regression_metrics(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Same as regression_metrics, but at the batch level. For aggregation purposes.
    """
    return regression_metrics(pred_raw, target_raw, mask)