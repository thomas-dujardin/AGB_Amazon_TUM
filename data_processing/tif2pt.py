import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

import torch
from tqdm import tqdm


INPUT_BANDS = ["VV", "VH", "B02", "B03", "B04", "B08", "B11", "B12"]
TARGET_BANDS = ["AGB", "SD", "valid_mask"]


def read_tif(path):
    with rasterio.open(path) as src:
        array = src.read().astype(np.float32)  # [C, H, W]
        info = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "dtypes": src.dtypes,
            "nodatavals": src.nodatavals,
        }

    return array, info


def lon_lat_from_bounds(bounds, crs):
    cx = 0.5 * (bounds.left + bounds.right)
    cy = 0.5 * (bounds.bottom + bounds.top)

    if crs is None:
        return float(cx), float(cy)

    if crs.to_string() == "EPSG:4326":
        return float(cx), float(cy)

    lon, lat = warp_transform(crs, "EPSG:4326", [cx], [cy])
    return float(lon[0]), float(lat[0])


def find_tile_ids(input_dir):
    input_dir = Path(input_dir)
    tile_ids = []

    for img_path in input_dir.glob("img_tile_*.tif"):
        tile_id = int(img_path.stem.split("_")[-1])
        esa_path = input_dir / f"esa_tile_{tile_id}.tif"

        if esa_path.exists():
            tile_ids.append(tile_id)

    return sorted(tile_ids)

def clean_agb_and_sd(agb, sd, valid_mask):
    valid_mask = (
        (valid_mask > 0.5)
        & np.isfinite(agb)
        & (agb != -9999)
    ).astype(np.float32)

    # AGB is physical biomass density and can exceed 256, it's not a grayscale image.
    # Do not divide by 255 or rescale per tile.
    agb = np.nan_to_num(agb, nan=0.0, posinf=0.0, neginf=0.0)
    agb = np.where(agb == -9999, 0.0, agb).astype(np.float32)

    sd = np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)
    sd = np.where(valid_mask > 0.5, sd, 0.0).astype(np.float32)

    return agb, sd, valid_mask


def convert_tile(tile_id, input_dir, output_dir):
    img_path = input_dir / f"img_tile_{tile_id}.tif"
    esa_path = input_dir / f"esa_tile_{tile_id}.tif"

    img, img_info = read_tif(img_path)
    esa, esa_info = read_tif(esa_path)

    # img_tile_i.tif:
    #   [VV, VH, B02, B03, B04, B08, B11, B12]
    x = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # esa_tile_i.tif:
    #   [AGB, SD, valid_mask]
    agb = esa[0:1]
    sd = esa[1:2]
    valid_mask = esa[2:3]

    agb, sd, valid_mask = clean_agb_and_sd(agb, sd, valid_mask)

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(agb).float()
    sd = torch.from_numpy(sd).float()
    valid_mask = torch.from_numpy(valid_mask).float()

    lon, lat = lon_lat_from_bounds(img_info["bounds"], img_info["crs"])

    # [lon, lat, time, scale]
    # time = 0 for annual composite, scale = 10 because input is on 10 m grid.
    meta = torch.tensor([lon, lat, 0.0, 10.0], dtype=torch.float32)

    tile_dir = output_dir / f"tile_{tile_id:06d}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    torch.save(x, tile_dir / "x.pt")
    torch.save(y, tile_dir / "y.pt")
    torch.save(sd, tile_dir / "sd.pt")
    torch.save(valid_mask, tile_dir / "valid_mask.pt")
    torch.save(meta, tile_dir / "meta.pt")

    with open(tile_dir / "bands.json", "w") as f:
        json.dump(
            {
                "tile_id": tile_id,
                "input_band_names": INPUT_BANDS,
                "target_band_names": TARGET_BANDS,
                "x_shape": list(x.shape),
                "y_shape": list(y.shape),
                "sd_shape": list(sd.shape),
                "valid_mask_shape": list(valid_mask.shape),
                "input_crs": img_info["crs"].to_string() if img_info["crs"] else None,
                "target_crs": esa_info["crs"].to_string() if esa_info["crs"] else None,
                "input_transform": list(img_info["transform"]),
                "target_transform": list(esa_info["transform"]),
                "input_bounds": list(img_info["bounds"]),
                "target_bounds": list(esa_info["bounds"]),
                "centroid_lon": lon,
                "centroid_lat": lat,
                "input_resolution_m": 10,
                "target_resolution_m": 100,
            },
            f,
            indent=2,
        )

    return {
        "tile_id": tile_id,
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "sd_shape": list(sd.shape),
        "valid_pixels": int(valid_mask.sum().item()),
        "agb_min": float(y[valid_mask.bool()].min().item()) if valid_mask.sum() > 0 else None,
        "agb_max": float(y[valid_mask.bool()].max().item()) if valid_mask.sum() > 0 else None,
        "agb_mean": float(y[valid_mask.bool()].mean().item()) if valid_mask.sum() > 0 else None,
        "sd_mean": float(sd[valid_mask.bool()].mean().item()) if valid_mask.sum() > 0 else None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="biomass_tiles_320_annual_agb_sd_corrected",
        help="Folder containing img_tile_i.tif and esa_tile_i.tif files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="biomass_tiles_320_pt",
        help="Folder where tile_000000/x.pt, y.pt, sd.pt, etc. will be written.",
    )
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tile_ids = find_tile_ids(input_dir)

    if args.start_index is not None:
        tile_ids = [i for i in tile_ids if i >= args.start_index]

    if args.end_index is not None:
        tile_ids = [i for i in tile_ids if i < args.end_index]

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for tile_id in tqdm(tile_ids, desc="Converting tiles"):
        row = convert_tile(tile_id, input_dir, output_dir)
        rows.append(row)

    summary = {
        "num_tiles": len(tile_ids),
        "tile_ids": tile_ids,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_bands": INPUT_BANDS,
        "target_bands": TARGET_BANDS,
        "expected_x_shape": [8, 320, 320],
        "expected_y_shape": [1, 32, 32],
        "expected_sd_shape": [1, 32, 32],
        "expected_valid_mask_shape": [1, 32, 32],
        "tiles": rows,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()