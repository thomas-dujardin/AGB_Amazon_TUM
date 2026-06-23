import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from biomass.data import (
    BiomassTileDataset,
    inverse_target_transform,
    list_tile_ids,
    prepare_cache_dir,
)
from biomass.losses import masked_l1, masked_rmse
from biomass.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description="Run inference with a trained Copernicus-FM biomass model.",
    )

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--hf_subdir", type=str, default="biomass_tiles_320_pt")
    parser.add_argument("--hf_local_dir", type=str, default="data/hf")

    parser.add_argument("--tile_id", type=int, default=None)
    parser.add_argument("--all_tiles", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to best.pt produced by scripts/train_eval.py.",
    )
    parser.add_argument(
        "--copernicus_checkpoint",
        type=str,
        default=None,
        help="Optional override for the Copernicus-FM checkpoint path.",
    )

    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    parser.add_argument("--save_pred_pt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_csv", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_gpu", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def get_device(use_gpu: bool) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_training_checkpoint(path: str, device: torch.device) -> Dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    return torch.load(path, map_location=device)


def build_model_from_checkpoint(ckpt: Dict, args, device: torch.device):
    train_args = ckpt.get("args", {})

    copernicus_checkpoint = args.copernicus_checkpoint
    if copernicus_checkpoint is None:
        copernicus_checkpoint = train_args.get(
            "checkpoint_path",
            "checkpoints/CopernicusFM_ViT_base_varlang_e100.pth",
        )

    model = build_model(
        checkpoint_path=copernicus_checkpoint,
        random_init_copernicus=bool(train_args.get("random_init_copernicus", False)),
        train_encoder=False,
        decoder_hidden_dim=int(train_args.get("decoder_hidden_dim", 256)),
        decoder_dropout=float(train_args.get("decoder_dropout", 0.0)),
        refiner_on=bool(train_args.get("refiner_on", False)),
        refiner_width=int(train_args.get("refiner_width", 64)),
        refiner_dropout=float(train_args.get("refiner_dropout", 0.0)),
        out_channels=1,
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model


def select_tile_ids(cache_dir: Path, tile_id: Optional[int], all_tiles: bool) -> List[int]:
    if tile_id is not None:
        return [int(tile_id)]

    if all_tiles:
        return list_tile_ids(cache_dir)

    raise ValueError("Please pass either --tile_id <id> or --all_tiles.")


def normalize_for_png(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    finite = torch.isfinite(x)

    if finite.sum() < 2:
        return torch.zeros_like(x)

    vals = x[finite]
    lo = torch.quantile(vals, 0.02)
    hi = torch.quantile(vals, 0.98)

    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def save_prediction_panel(
    output_path: Path,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    sd: torch.Tensor,
    valid_mask: torch.Tensor,
):
    """
    Saves:
        target | prediction | absolute error | SD | valid mask

    The true tensors are 32x32. The 320x320 version is only for visualization.
    """
    pred = pred_raw[0:1]
    target = target_raw[0:1]
    sd = sd[0:1]
    mask = valid_mask[0:1]

    error = torch.abs(pred - target)

    maps = [
        normalize_for_png(target),
        normalize_for_png(pred),
        normalize_for_png(error),
        normalize_for_png(sd),
        mask.detach().float().cpu().clamp(0, 1),
    ]

    maps = [
        F.interpolate(m.unsqueeze(0), size=(320, 320), mode="nearest")[0]
        for m in maps
    ]

    panel = torch.cat(maps, dim=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(panel, output_path)


def metric_row(
    tile_id: int,
    meta: torch.Tensor,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    mask = valid_mask.float()

    mae = masked_l1(pred_raw, target_raw, mask)
    rmse = masked_rmse(pred_raw, target_raw, mask)

    valid_pixels = int(mask.sum().item())
    diff = (pred_raw - target_raw) * mask

    if valid_pixels > 0:
        bias = float(diff.sum().item() / valid_pixels)
        pred_mean = float((pred_raw * mask).sum().item() / valid_pixels)
        target_mean = float((target_raw * mask).sum().item() / valid_pixels)
    else:
        bias = float("nan")
        pred_mean = float("nan")
        target_mean = float("nan")

    meta = meta.detach().cpu()

    return {
        "tile_id": int(tile_id),
        "lon": float(meta[0].item()),
        "lat": float(meta[1].item()),
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "bias": bias,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "valid_pixels": valid_pixels,
    }


def write_csv(rows: List[Dict], path: Path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(obj: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def summarize_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {}

    maes = np.array([r["mae"] for r in rows], dtype=np.float64)
    rmses = np.array([r["rmse"] for r in rows], dtype=np.float64)
    biases = np.array([r["bias"] for r in rows], dtype=np.float64)

    return {
        "num_tiles": len(rows),
        "mae_mean": float(np.nanmean(maes)),
        "rmse_mean": float(np.nanmean(rmses)),
        "bias_mean": float(np.nanmean(biases)),
        "mae_median": float(np.nanmedian(maes)),
        "rmse_median": float(np.nanmedian(rmses)),
    }


@torch.no_grad()
def run_inference():
    args = parse_args()
    device = get_device(args.use_gpu)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = load_training_checkpoint(args.model_checkpoint, device=device)
    train_args = ckpt.get("args", {})
    target_stats = ckpt["target_stats"]

    target_transform = train_args.get("target_transform", "log1p")
    standardize_target = bool(train_args.get("standardize_target", True))
    normalize_input = bool(train_args.get("normalize_input", True))

    cache_dir = prepare_cache_dir(
        cache_dir=args.cache_dir,
        hf_dataset=args.hf_dataset,
        hf_subdir=args.hf_subdir,
        hf_local_dir=args.hf_local_dir,
    )
    cache_dir = Path(cache_dir)

    tile_ids = select_tile_ids(
        cache_dir=cache_dir,
        tile_id=args.tile_id,
        all_tiles=args.all_tiles,
    )

    print(f"Device: {device}")
    print(f"Using cache_dir: {cache_dir}")
    print(f"Number of tiles: {len(tile_ids)}")
    print(f"Target transform: {target_transform}")
    print(f"Standardize target: {standardize_target}")
    print(f"Normalize input: {normalize_input}")

    dataset = BiomassTileDataset(
        tile_ids=tile_ids,
        cache_dir=cache_dir,
        target_stats=target_stats,
        target_transform=target_transform,
        standardize_target=standardize_target,
        normalize_input=normalize_input,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model_from_checkpoint(
        ckpt=ckpt,
        args=args,
        device=device,
    )

    rows = []

    for batch in tqdm(loader, desc="Inference"):
        image = batch["image"].to(device, non_blocking=True)
        meta = batch["meta"].to(device, non_blocking=True)

        pred = model(image, meta)

        pred_raw = inverse_target_transform(
            pred,
            mean=target_stats["mean"],
            std=target_stats["std"],
            transform=target_transform,
            standardized=standardize_target,
        )

        pred_raw = pred_raw.detach().cpu()
        target_raw = batch["target_raw"].detach().cpu()
        sd = batch["sd"].detach().cpu()
        valid_mask = batch["valid_mask"].detach().cpu()
        meta_cpu = batch["meta"].detach().cpu()
        tile_id_cpu = batch["tile_id"].detach().cpu()

        for i in range(pred_raw.size(0)):
            tile_id = int(tile_id_cpu[i].item())

            tile_out = output_dir / f"tile_{tile_id:06d}"
            tile_out.mkdir(parents=True, exist_ok=True)

            if args.save_pred_pt:
                torch.save(pred_raw[i], tile_out / "pred_agb.pt")

            if args.save_png:
                save_prediction_panel(
                    output_path=tile_out / "panel_target_pred_error_sd_mask.png",
                    pred_raw=pred_raw[i],
                    target_raw=target_raw[i],
                    sd=sd[i],
                    valid_mask=valid_mask[i],
                )

            rows.append(
                metric_row(
                    tile_id=tile_id,
                    meta=meta_cpu[i],
                    pred_raw=pred_raw[i:i + 1],
                    target_raw=target_raw[i:i + 1],
                    valid_mask=valid_mask[i:i + 1],
                )
            )

    summary = summarize_rows(rows)

    if args.save_csv:
        write_csv(rows, output_dir / "inference_per_tile.csv")

    write_json(summary, output_dir / "inference_summary.json")

    print("\nInference summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved outputs to:")
    print(output_dir)


if __name__ == "__main__":
    run_inference()