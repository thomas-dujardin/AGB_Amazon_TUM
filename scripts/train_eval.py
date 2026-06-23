#!/usr/bin/env python
# Main training script for CFM-v1 on ESA CCI AGB prediction.

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from biomass.data import (
    inverse_target_transform,
    make_datasets,
    prepare_cache_dir,
)
from biomass.losses import make_loss
from biomass.models import build_model, check_forward_320


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train_eval.py",
        description="Train and evaluate Copernicus-FM for ESA CCI AGB prediction.",
    )
    # ---------------------------------------------------------------------
    # arguments related to data and dataset splitting
    # ---------------------------------------------------------------------
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--hf_subdir", type=str, default="biomass_tiles_320_pt")
    parser.add_argument("--hf_local_dir", type=str, default="data/hf")

    parser.add_argument("--split_mode", type=str, default="spatial", choices=["spatial", "random", "spatial_blocks"], help="Dataset split mode: random tile split, 1D spatial split, or spatial block split.")
    parser.add_argument("--spatial_axis", type=str, default="lon", choices=["lon", "lat"])
    parser.add_argument("--train_fraction", type=float, default=0.8) # 384 samples, don't make it too large
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--subsample_fraction", type=float, default=1.0) # Don't use that

    parser.add_argument("--spatial_grid_rows", type=int, default=4, help="Number of latitude bins for spatial_blocks split.")
    parser.add_argument("--spatial_grid_cols", type=int, default=4, help="Number of longitude bins for spatial_blocks split.")

    parser.add_argument("--target_transform", type=str, default="log1p", choices=["log1p", "none"])
    parser.add_argument("--standardize_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize_input", action=argparse.BooleanOptionalAction, default=True)

    # ---------------------------------------------------------------------
    # arguments related to model architecture and initialization
    # ---------------------------------------------------------------------
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/CopernicusFM_ViT_base_varlang_e100.pth")
    parser.add_argument("--random_init_copernicus", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_encoder", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_dropout", type=float, default=0.0)

    parser.add_argument("--refiner_on", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refiner_width", type=int, default=64)
    parser.add_argument("--refiner_dropout", type=float, default=0.0)

    # ---------------------------------------------------------------------
    # arguments related to loss function and weighting
    # ---------------------------------------------------------------------
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l1",
        choices=["l1", "mse", "sd_weighted_l1", "sd_weighted_mse"],
    )
    parser.add_argument("--laplacian_coeff", type=float, default=0.0)
    parser.add_argument("--sd_weight_power", type=float, default=0.5)
    parser.add_argument("--sd_weight_clip", type=float, default=10.0)
    parser.add_argument("--sd_logspace", action=argparse.BooleanOptionalAction, default=True)

    # ---------------------------------------------------------------------
    # arguments related to training hyperparameters
    # ---------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)

    # ---------------------------------------------------------------------
    # arguments related to logging, evaluation, and checkpointing
    # ---------------------------------------------------------------------
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--experiment_name", type=str, default="baseline")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_latest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_viz_every", type=int, default=1)
    parser.add_argument("--num_viz", type=int, default=2)

    parser.add_argument(
        "--monitor",
        type=str,
        default="val_rmse",
        choices=["val_loss", "val_mae", "val_rmse"],
    )

    parser.add_argument("--check_forward", action=argparse.BooleanOptionalAction, default=False)

    return parser.parse_args()

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(use_gpu: bool) -> torch.device:
    """Get the torch device to use for training/evaluation. No multi-GPU support for now."""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def make_run_dir(args) -> Path:
    """Logs for tensorboard, checkpoints, and outputs will be saved under this directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / args.experiment_name / timestamp
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    return run_dir

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def make_loaders(train_ds, val_ds, test_ds, args):
    persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader

def make_optimizer(model: nn.Module, args):
    """Initialize the optimizer with separate parameter groups for encoder and decoder/refiner if needed."""
    encoder_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            other_params.append(p)

    param_groups = []

    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": args.lr,
                "name": "decoder_refiner",
            }
        )

    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": args.encoder_lr,
                "name": "encoder",
            }
        )

    # Defaults to AdamW
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    return optimizer

def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Displays current lr, for tensorboard/logging purposes."""
    return optimizer.param_groups[0]["lr"]

def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a batch of tensors to the specified device."""
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out

def inverse_predictions(pred: torch.Tensor, batch: Dict[str, torch.Tensor], target_stats: Dict[str, float], args) -> torch.Tensor:
    """Convert model predictions back to raw target space from log1p for metric calculation and visualization."""

    mean, std = target_stats["mean"], target_stats["std"]

    pred_raw = inverse_target_transform(
        pred,
        mean=mean,
        std=std,
        transform=args.target_transform,
        standardized=args.standardize_target,
    )

    return pred_raw

def metric_sums_from_batch(pred_raw: torch.Tensor, target_raw: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, float]:
    """Calculate intermediate batch sums"""

    mask = valid_mask.float()
    n = float(mask.sum().item())

    if n <= 0:
        return {
            "n": 0.0,
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "bias_sum": 0.0,
            "pred_sum": 0.0,
            "target_sum": 0.0,
        }

    diff = (pred_raw - target_raw) * mask

    return {
        "n": n,
        "abs_sum": float(diff.abs().sum().item()),
        "sq_sum": float((diff ** 2).sum().item()),
        "bias_sum": float(diff.sum().item()),
        "pred_sum": float((pred_raw * mask).sum().item()),
        "target_sum": float((target_raw * mask).sum().item()),
    }

def merge_metric_sums(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Sums/merges metrics from batches and/or tiles for final metric calculations."""

    out = {
        "n": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "bias_sum": 0.0,
        "pred_sum": 0.0,
        "target_sum": 0.0,
    }

    for r in rows:
        for k in out:
            out[k] += r[k]

    return out

def finalize_metrics(sums: Dict[str, float]) -> Dict[str, float]:
    """Final metrics for the output folder"""

    n = max(sums["n"], 1.0)

    return {
        "mae": sums["abs_sum"] / n,
        "rmse": float(np.sqrt(sums["sq_sum"] / n)),
        "bias": sums["bias_sum"] / n,
        "pred_mean": sums["pred_sum"] / n,
        "target_mean": sums["target_sum"] / n,
        "valid_pixels": int(sums["n"]),
    }

def normalize_for_tb(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor for visualization in TensorBoard."""

    x = x.detach().float().cpu()
    finite = torch.isfinite(x)

    if finite.sum() < 2:
        return torch.zeros_like(x)

    vals = x[finite]
    lo = torch.quantile(vals, 0.02)
    hi = torch.quantile(vals, 0.98)

    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

def make_viz_panel(pred_raw: torch.Tensor, target_raw: torch.Tensor, sd: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns a single panel:
        target | prediction | absolute error | SD | valid mask

    32x32 upscaled to 320x320for viz purposes.
    """

    pred = pred_raw[0:1]
    target = target_raw[0:1]
    sd = sd[0:1]
    mask = valid_mask[0:1]

    err = torch.abs(pred - target)

    maps = [
        normalize_for_tb(target),
        normalize_for_tb(pred),
        normalize_for_tb(err),
        normalize_for_tb(sd),
        mask.detach().float().cpu().clamp(0, 1),
    ]

    maps = [
        F.interpolate(m.unsqueeze(0), size=(320, 320), mode="nearest")[0]
        for m in maps
    ]

    return torch.cat(maps, dim=2)

@torch.no_grad()
def log_visuals(writer: SummaryWriter, model: nn.Module, loader: DataLoader, device: torch.device, target_stats: Dict[str, float], args, tag: str, step: int):
    """Logs a few viz to TensorBoard for qualitative evaluation"""

    model.eval() # We're not training here

    logged = 0

    for batch in loader:
        batch = to_device(batch, device)

        pred = model(batch["image"], batch["meta"])
        pred_raw = inverse_predictions(pred, batch, target_stats, args)

        bsz = pred.size(0)

        for i in range(bsz):
            if logged >= args.num_viz:
                return

            panel = make_viz_panel(
                pred_raw=pred_raw[i],
                target_raw=batch["target_raw"][i],
                sd=batch["sd"][i],
                valid_mask=batch["valid_mask"][i],
            )

            tile_id = int(batch["tile_id"][i].detach().cpu().item())
            writer.add_image(f"{tag}/tile_{tile_id}", panel, step)

            logged += 1


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler, device: torch.device, args, writer: SummaryWriter, epoch: int, global_step: int) -> int:
    """Train the model for one epoch and log training metrics to TensorBoard."""

    model.train()

    running_loss = 0.0
    running_n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

    for batch in pbar:
        batch = to_device(batch, device)

        pred = model(batch["image"], batch["meta"])

        #Use our loss fcts to compute the total loss. Adapts to both logspace and non-logspace.

        loss_dict = make_loss(
            pred=pred,
            target=batch["target"],
            target_raw=batch["target_raw"],
            sd=batch["sd"],
            mask=batch["valid_mask"],
            loss_type=args.loss_type,
            laplacian_coeff=args.laplacian_coeff,
            sd_weight_power=args.sd_weight_power,
            sd_weight_clip=args.sd_weight_clip,
            sd_logspace=args.sd_logspace,
        )

        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        bsz = batch["image"].size(0)
        running_loss += float(loss.item()) * bsz
        running_n += bsz

        if global_step % args.log_every == 0:
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/main_loss", float(loss_dict["main"].item()), global_step)
            writer.add_scalar("train/laplacian", float(loss_dict["laplacian"].item()), global_step)
            writer.add_scalar("train/lr", current_lr(optimizer), global_step)

        global_step += 1

        pbar.set_postfix(
            loss=running_loss / max(running_n, 1),
            lr=current_lr(optimizer),
        )

    return global_step


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, target_stats: Dict[str, float], args, split_name: str, save_per_tile: bool = False) -> Dict[str, object]:
    """Evaluate the model on the val split"""

    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_samples = 0

    metric_rows = []
    per_tile_rows = []

    for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
        batch = to_device(batch, device)

        pred = model(batch["image"], batch["meta"])

        # Same loss as in training, but we keep track of the raw predictions and targets for metric calculation and visualization. Adapts to both logspace and non-logspace.
        loss_dict = make_loss(
            pred=pred,
            target=batch["target"],
            target_raw=batch["target_raw"],
            sd=batch["sd"],
            mask=batch["valid_mask"],
            loss_type=args.loss_type,
            laplacian_coeff=args.laplacian_coeff,
            sd_weight_power=args.sd_weight_power,
            sd_weight_clip=args.sd_weight_clip,
            sd_logspace=args.sd_logspace,
        )

        bsz = batch["image"].size(0)
        total_loss += float(loss_dict["loss"].item()) * bsz
        total_samples += bsz
        total_batches += 1

        pred_raw = inverse_predictions(pred, batch, target_stats, args)

        metric_rows.append(
            metric_sums_from_batch(
                pred_raw=pred_raw,
                target_raw=batch["target_raw"],
                valid_mask=batch["valid_mask"],
            )
        )

        if save_per_tile:
            for i in range(bsz):
                tile_id = int(batch["tile_id"][i].detach().cpu().item())

                sums_i = metric_sums_from_batch(
                    pred_raw=pred_raw[i:i + 1],
                    target_raw=batch["target_raw"][i:i + 1],
                    valid_mask=batch["valid_mask"][i:i + 1],
                )
                metrics_i = finalize_metrics(sums_i)

                meta_i = batch["meta"][i].detach().cpu()
                per_tile_rows.append(
                    {
                        "tile_id": tile_id,
                        "lon": float(meta_i[0].item()),
                        "lat": float(meta_i[1].item()),
                        "mae": metrics_i["mae"],
                        "rmse": metrics_i["rmse"],
                        "bias": metrics_i["bias"],
                        "pred_mean": metrics_i["pred_mean"],
                        "target_mean": metrics_i["target_mean"],
                        "valid_pixels": metrics_i["valid_pixels"],
                    }
                )

    sums = merge_metric_sums(metric_rows)
    metrics = finalize_metrics(sums)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["num_samples"] = int(total_samples)
    metrics["num_batches"] = int(total_batches)

    out = {
        "metrics": metrics,
        "per_tile": per_tile_rows,
    }

    return out

def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, best_score: float, target_stats: Dict[str, float], split_info: Dict[str, object], args, global_step: int):
    """Saves a given checkpoint for later (usually the best checkpoint on the val split)"""

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "target_stats": target_stats,
        "split_info": split_info,
        "args": vars(args),
        "global_step": global_step,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: Path, model: nn.Module, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt

def write_per_tile_csv(rows: List[Dict[str, object]], path: Path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def metric_to_monitor(eval_out: Dict[str, object], monitor: str) -> float:
    metrics = eval_out["metrics"]

    if monitor == "val_loss":
        return float(metrics["loss"])
    if monitor == "val_mae":
        return float(metrics["mae"])
    if monitor == "val_rmse":
        return float(metrics["rmse"])

def main():
    args = parse_args()
    seed_everything(args.seed)

    device = get_device(args.use_gpu)
    run_dir = make_run_dir(args)

    save_json(vars(args), run_dir / "args.json")

    cache_dir = prepare_cache_dir(
        cache_dir=args.cache_dir,
        hf_dataset=args.hf_dataset,
        hf_subdir=args.hf_subdir,
        hf_local_dir=args.hf_local_dir,
    )

    print(f"The HF cache dir, if it exists, is right here: {cache_dir}")

    train_ds, val_ds, test_ds, split_info = make_datasets(
        cache_dir=cache_dir,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        spatial_axis=args.spatial_axis,
        subsample_fraction=args.subsample_fraction,
        target_transform=args.target_transform,
        standardize_target=args.standardize_target,
        normalize_input=args.normalize_input,
        spatial_grid_rows=args.spatial_grid_rows,
        spatial_grid_cols=args.spatial_grid_cols,
    )

    save_json(split_info, run_dir / "split_info.json")

    target_stats = split_info["target_stats"]

    print("General info about the dataset and splits:")
    print("Dataset sizes:")
    print(f"  train: {len(train_ds)}")
    print(f"  val:   {len(val_ds)}")
    print(f"  test:  {len(test_ds)}")
    print("Target stats:")
    print(json.dumps(target_stats, indent=2))

    train_loader, val_loader, test_loader = make_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        args=args,
    )

    model = build_model(
        checkpoint_path=args.checkpoint_path,
        random_init_copernicus=args.random_init_copernicus,
        train_encoder=args.train_encoder,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_dropout=args.decoder_dropout,
        refiner_on=args.refiner_on,
        refiner_width=args.refiner_width,
        refiner_dropout=args.refiner_dropout,
        out_channels=1,
    )

    model = model.to(device)

    if args.check_forward:
        out = check_forward_320(model, device=device, batch_size=1)

    optimizer = make_optimizer(model, args)

    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.eta_min,
    )

    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"TensorBoard:")
    print(f"  tensorboard --logdir {args.run_root}")

    best_score = float("inf")
    best_path = run_dir / "checkpoints" / "best.pt"
    latest_path = run_dir / "checkpoints" / "latest.pt"

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args=args,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
        )

        if epoch % args.eval_every == 0:
            val_out = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                target_stats=target_stats,
                args=args,
                split_name="val",
                save_per_tile=False,
            )

            val_metrics = val_out["metrics"]

            writer.add_scalar("val/loss", val_metrics["loss"], epoch)
            writer.add_scalar("val/mae_raw", val_metrics["mae"], epoch)
            writer.add_scalar("val/rmse_raw", val_metrics["rmse"], epoch)
            writer.add_scalar("val/bias_raw", val_metrics["bias"], epoch)
            writer.add_scalar("val/pred_mean_raw", val_metrics["pred_mean"], epoch)
            writer.add_scalar("val/target_mean_raw", val_metrics["target_mean"], epoch)

            score = metric_to_monitor(val_out, args.monitor)

            print(f"You can check this metrics on TensorBoard, and locally in {run_dir / 'outputs' / 'val_metrics.json'}")

            if score < best_score:
                best_score = score

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=best_score,
                    target_stats=target_stats,
                    split_info=split_info,
                    args=args,
                    global_step=global_step,
                )

                print(f"New best checkpoint: {best_path}")

            if args.save_viz_every > 0 and epoch % args.save_viz_every == 0:
                log_visuals(
                    writer=writer,
                    model=model,
                    loader=val_loader,
                    device=device,
                    target_stats=target_stats,
                    args=args,
                    tag="val_images",
                    step=epoch,
                )

        if args.save_latest:
            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                target_stats=target_stats,
                split_info=split_info,
                args=args,
                global_step=global_step,
            )

    writer.flush()

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best {args.monitor}: {best_score:.6f}")

    print("\nBest checkpoint metrics on the test set:")
    best_ckpt = load_checkpoint(best_path, model, device=device)

    test_out = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        target_stats=best_ckpt["target_stats"],
        args=args,
        split_name="test",
        save_per_tile=True,
    )

    test_metrics = test_out["metrics"]

    save_json(test_metrics, run_dir / "outputs" / "test_metrics.json")
    write_per_tile_csv(test_out["per_tile"], run_dir / "outputs" / "test_per_tile.csv")

    writer.add_scalar("test/loss", test_metrics["loss"], args.epochs)
    writer.add_scalar("test/mae_raw", test_metrics["mae"], args.epochs)
    writer.add_scalar("test/rmse_raw", test_metrics["rmse"], args.epochs)
    writer.add_scalar("test/bias_raw", test_metrics["bias"], args.epochs)

    log_visuals(
        writer=writer,
        model=model,
        loader=test_loader,
        device=device,
        target_stats=best_ckpt["target_stats"],
        args=args,
        tag="test_images",
        step=args.epochs,
    )

    writer.close()

    print("\nTest split metrics for the best checkpoint:")
    print(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()