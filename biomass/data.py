import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


INPUT_BANDS = ["VV", "VH", "B02", "B03", "B04", "B08", "B11", "B12"]
TARGET_BANDS = ["AGB", "SD", "valid_mask"]


def prepare_cache_dir(cache_dir: Optional[str] = None, hf_dataset: Optional[str] = None, hf_subdir: str = "biomass_tiles_320_pt", hf_local_dir: str = "data/hf") -> Path:
    """
    Returns a local folder containing tile_XXXXXX folders.

    Two possible modes:

    1. Local mode:
        --cache_dir data/biomass_tiles_320_pt

    2. Hugging Face mode:
        --hf_dataset tdujardin/amazon_basin_pt_320
        --hf_subdir biomass_tiles_320_pt
        --hf_local_dir data/hf

    In HF mode, the dataset is downloaded once by huggingface_hub. Later runs reuse
    the local folder.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

        return cache_dir

    from huggingface_hub import snapshot_download

    hf_local_dir = Path(hf_local_dir)
    hf_local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=hf_dataset,
        repo_type="dataset",
        allow_patterns=[f"{hf_subdir}/**"],
        local_dir=str(hf_local_dir),
    )

    cache_dir = hf_local_dir / hf_subdir

    return cache_dir


def list_tile_ids(cache_dir: str | Path) -> List[int]:
    """Makes a list of tile IDs by looking at the cache directory if it is there"""
    cache_dir = Path(cache_dir)

    tile_ids = []

    for p in sorted(cache_dir.glob("tile_*")):
        if not p.is_dir():
            continue

        try:
            tile_id = int(p.name.split("_")[-1])
        except ValueError:
            continue

        required = [
            p / "x.pt",
            p / "y.pt",
            p / "sd.pt",
            p / "valid_mask.pt",
            p / "meta.pt",
            p / "bands.json",
        ]

        if all(q.exists() for q in required):
            tile_ids.append(tile_id)

    return sorted(tile_ids)


def tile_dir_from_id(cache_dir: str | Path, tile_id: int) -> Path:
    return Path(cache_dir) / f"tile_{int(tile_id):06d}"


def load_meta(cache_dir: str | Path, tile_id: int) -> torch.Tensor:
    path = tile_dir_from_id(cache_dir, tile_id) / "meta.pt"
    return torch.load(path, map_location="cpu").float()


def normalize_input_per_tile(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-channel normalization over H,W.
    The .pt cache stores physical/raw values. Normalization happens at dataset time to keep the cache clean.
    """

    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)

    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True)

    return (x - mean) / (std + eps)


def apply_target_transform(y_raw: torch.Tensor, transform: str) -> torch.Tensor:
    """
    y_raw is raw AGB. It is not grayscale and should never be divided by 255.
    """
    y_raw = torch.nan_to_num(y_raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
    y_raw = y_raw.clamp_min(0.0)

    if transform == "none":
        return y_raw

    if transform == "log1p":
        return torch.log1p(y_raw)


def inverse_target_transform(y: torch.Tensor, mean: torch.Tensor | float = 0.0, std: torch.Tensor | float = 1.0, transform: str = "log1p", standardized: bool = True) -> torch.Tensor:
    """
    Converts a model output back to raw AGB units.

    (log1p to expm1)

    Metrics should be computed in this raw AGB space.
    """
    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, dtype=y.dtype, device=y.device)
    else:
        mean = mean.to(device=y.device, dtype=y.dtype)

    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=y.dtype, device=y.device)
    else:
        std = std.to(device=y.device, dtype=y.dtype)

    out = y

    if standardized:
        out = out * std + mean

    if transform == "none":
        return out.clamp_min(0.0)

    if transform == "log1p":
        return torch.expm1(out).clamp_min(0.0)

def compute_target_stats(tile_ids: Sequence[int], cache_dir: str | Path, target_transform: str = "log1p") -> Dict[str, float]:
    """
    Computes target statistics on the training split only.

    These statistics are used to standardize the transformed target:
        y = (transform(y_raw) - mean) / std
    """
    values = []

    for tile_id in tile_ids:
        tile_dir = tile_dir_from_id(cache_dir, tile_id)

        y_raw = torch.load(tile_dir / "y.pt", map_location="cpu").float()
        valid_mask = torch.load(tile_dir / "valid_mask.pt", map_location="cpu").bool()

        y = apply_target_transform(y_raw, target_transform)

        v = y[valid_mask]
        v = v[torch.isfinite(v)]

        if v.numel() > 0:
            values.append(v)

    values = torch.cat(values)

    return {
        "mean": float(values.mean().item()),
        "std": float(values.std().clamp_min(1e-6).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "n": int(values.numel()),
        "target_transform": target_transform,
    }


def subsample_tile_ids(tile_ids: Sequence[int], fraction: float, seed: int, ) -> List[int]:
    """Subsampling, just in case the number of tiles gets bigger in the future"""
    tile_ids = np.array(list(tile_ids), dtype=np.int64)

    if fraction >= 1.0:
        return sorted(tile_ids.tolist())

    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(tile_ids) * fraction)))
    sampled = rng.choice(tile_ids, size=n, replace=False)

    return sorted(sampled.tolist())


def assign_spatial_blocks(tile_ids, root_dir: Path, rows: int = 4, cols: int = 4) -> Dict[int, int]:
    """
    Assign each tile to a spatial grid block using lon/lat from meta.pt.

    load_meta(root_dir, tile_id) is expected to return:
        [lon, lat, time, scale]

    Returns:
        dict[tile_id] = block_id
    """
    lons = []
    lats = []

    for tile_id in tile_ids:
        meta = load_meta(root_dir, tile_id)
        lon = float(meta[0])
        lat = float(meta[1])
        lons.append(lon)
        lats.append(lat)

    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    lon_min, lon_max = float(lons.min()), float(lons.max())
    lat_min, lat_max = float(lats.min()), float(lats.max())

    eps = 1e-12
    block_by_tile = {}

    for tile_id, lon, lat in zip(tile_ids, lons, lats):
        col = int((lon - lon_min) / max(lon_max - lon_min, eps) * cols)
        row = int((lat - lat_min) / max(lat_max - lat_min, eps) * rows)

        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)

        block_id = row * cols + col
        block_by_tile[int(tile_id)] = int(block_id)

    return block_by_tile


def split_tile_ids(tile_ids, root_dir: Path, train_fraction: float = 0.8, val_fraction: float = 0.1, split_mode: str = "spatial_blocks", seed: int = 42, spatial_axis: str = "lon", spatial_grid_rows: int = 4, spatial_grid_cols: int = 4) -> Tuple[List[int], List[int], List[int], Dict[str, object]]:
    """
    Split tile IDs into train/val/test.
    spatial_blocks split mode is recommanded.

    tile_ids: List of tile IDs to split.
    root_dir: Path to the HF cache directory, if it exists.
    train_fraction: Fraction of tiles to use for training.
    val_fraction: Fraction of tiles to use for validation. Has to be less than 1 - train_fraction.
    split_mode: One of "random", "spatial", "spatial_blocks":
        random:
            Random tile-level split (ie at the tile ID level).

        spatial:
            One-dimensional spatial split by lon or lat.

        spatial_blocks:
            Split whole spatial blocks into train/val/test.
            This is better for testing local spatial generalization.
    seed: Random seed for reproducibility.
    spatial_axis: If split_mode is "spatial", should the splitting happens horizontally (by "lon") or vertically (by "lat").
    spatial_grid_rows: If split_mode is "spatial_blocks", how many rows to use for the spatial grid.
    spatial_grid_cols: If split_mode is "spatial_blocks", how many columns to use for the spatial grid.
    """
    tile_ids = [int(t) for t in tile_ids]
    rng = np.random.default_rng(seed) # For SDs evaluations and random splits

    n_total = len(tile_ids)

    if split_mode == "random":
        ids = np.asarray(tile_ids)
        rng.shuffle(ids)

        n_train = int(round(train_fraction * n_total))
        n_val = int(round(val_fraction * n_total))

        train_ids = ids[:n_train].tolist()
        val_ids = ids[n_train:n_train + n_val].tolist()
        test_ids = ids[n_train + n_val:].tolist()

    elif split_mode == "spatial":
        coords = []

        for tile_id in tile_ids:
            meta = load_meta(root_dir, tile_id)
            lon = float(meta[0])
            lat = float(meta[1])

            if spatial_axis == "lon":
                key = lon
            else spatial_axis == "lat":
                key = lat

            coords.append((key, tile_id))

        coords = sorted(coords, key=lambda x: x[0])
        ordered_ids = [tile_id for _, tile_id in coords]

        n_train = int(round(train_fraction * n_total))
        n_val = int(round(val_fraction * n_total))

        train_ids = ordered_ids[:n_train]
        val_ids = ordered_ids[n_train:n_train + n_val]
        test_ids = ordered_ids[n_train + n_val:]

    elif split_mode == "spatial_blocks":
        block_by_tile = assign_spatial_blocks(
            tile_ids=tile_ids,
            root_dir=root_dir,
            rows=spatial_grid_rows,
            cols=spatial_grid_cols,
        )

        unique_blocks = sorted(set(block_by_tile.values()))
        unique_blocks = np.asarray(unique_blocks, dtype=np.int64)
        rng.shuffle(unique_blocks)

        n_blocks = len(unique_blocks)

        n_train_blocks = int(round(train_fraction * n_blocks))
        n_val_blocks = int(round(val_fraction * n_blocks))

        n_train_blocks = max(1, min(n_train_blocks, n_blocks - 2))
        n_val_blocks = max(1, min(n_val_blocks, n_blocks - n_train_blocks - 1))

        train_blocks = set(unique_blocks[:n_train_blocks].tolist())
        val_blocks = set(
            unique_blocks[n_train_blocks:n_train_blocks + n_val_blocks].tolist()
        )
        test_blocks = set(unique_blocks[n_train_blocks + n_val_blocks:].tolist())

        train_ids = [
            tile_id for tile_id in tile_ids
            if block_by_tile[tile_id] in train_blocks
        ]
        val_ids = [
            tile_id for tile_id in tile_ids
            if block_by_tile[tile_id] in val_blocks
        ]
        test_ids = [
            tile_id for tile_id in tile_ids
            if block_by_tile[tile_id] in test_blocks
        ]

    # everything required for the train/val/test split is in this dict.
    split_info = {
        "split_mode": split_mode,
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "test_fraction": float(1.0 - train_fraction - val_fraction),
        "spatial_axis": spatial_axis,
        "spatial_grid_rows": int(spatial_grid_rows),
        "spatial_grid_cols": int(spatial_grid_cols),
        "n_total": int(n_total),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "train_ids": [int(x) for x in train_ids],
        "val_ids": [int(x) for x in val_ids],
        "test_ids": [int(x) for x in test_ids],
    }

    if split_mode == "spatial_blocks": # which tile belongs to which spatial block.
        split_info["block_by_tile"] = {
            str(k): int(v) for k, v in block_by_tile.items()
        }

    return train_ids, val_ids, test_ids, split_info

class BiomassTileDataset(Dataset):
    """
    Dataset for the selected Amazon Basin AOI .pt cache.
    Example tile folder structure:
        tile_000000/
            x.pt              [8, 320, 320]
            y.pt              [1, 32, 32] raw AGB
            sd.pt             [1, 32, 32] raw ESA CCI SD (SD as in standard deviation)
            valid_mask.pt     [1, 32, 32] mask for losses and metrics
            meta.pt           [lon, lat, time, scale] 
            bands.json        metadata required by the CFM-v1 model

    Returns:

        image:
            [8, 320, 320], stacking bands to create a CFM-v1 input (VV, VH, B02, B03, B04, B08, B11, B12 from sentinel 1 and 2)

        target:
            [1, 32, 32], transformed and optionally standardized target (ESA CCI AGB has a resol)

        target_raw:
            [1, 32, 32], raw AGB values

        sd:
            [1, 32, 32], ESA CCI uncertainty

        valid_mask:
            [1, 32, 32], mask for losses and metrics

        meta:
            [4], Copernicus-FM metadata: [lon, lat, time, scale]

        tile_id:
            integer tile id
    """
    def __init__(self, tile_ids: Sequence[int], cache_dir: str | Path, target_stats: Optional[Dict[str, float]] = None, target_transform: str = "log1p", standardize_target: bool = True, normalize_input: bool = True, input_band_order: Optional[Iterable[str]] = None):
        """Initialize required objects"""
        self.tile_ids = [int(x) for x in tile_ids]
        self.cache_dir = Path(cache_dir)

        self.target_stats = target_stats
        self.target_transform = target_transform
        self.standardize_target = standardize_target
        self.normalize_input = normalize_input

        self.input_band_order = (
            list(input_band_order)
            if input_band_order is not None
            else list(INPUT_BANDS)
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _band_indices(self, tile_dir: Path) -> List[int]:
        with open(tile_dir / "bands.json", "r") as f:
            info = json.load(f)

        names = info["input_band_names"]
        name_to_idx = {name: i for i, name in enumerate(names)}

        aliases = {
            "VV": ["VV", "S1_VV"],
            "VH": ["VH", "S1_VH"],
            "B02": ["B02", "B2"],
            "B03": ["B03", "B3"],
            "B04": ["B04", "B4"],
            "B08": ["B08", "B8"],
            "B11": ["B11"],
            "B12": ["B12"],
        }

        indices = []

        for band in self.input_band_order:
            found = found = name_to_idx[aliases.get(band, [band])]:

            indices.append(found)

        return indices

    def _standardize_target(self, y: torch.Tensor) -> torch.Tensor:
        if not self.standardize_target:
            return y

        mean = float(self.target_stats["mean"])
        std = float(self.target_stats["std"])

        return (y - mean) / (std + 1e-6)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile_id = int(self.tile_ids[idx])
        tile_dir = tile_dir_from_id(self.cache_dir, tile_id)

        x = torch.load(tile_dir / "x.pt", map_location="cpu").float()
        y_raw = torch.load(tile_dir / "y.pt", map_location="cpu").float()
        sd = torch.load(tile_dir / "sd.pt", map_location="cpu").float()
        valid_mask = torch.load(tile_dir / "valid_mask.pt", map_location="cpu").float()
        meta = torch.load(tile_dir / "meta.pt", map_location="cpu").float()

        x = x[self._band_indices(tile_dir)]

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize_input:
            x = normalize_input_per_tile(x)

        valid_mask = (valid_mask > 0.5).float()

        y_raw = torch.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
        y_raw = y_raw.clamp_min(0.0)

        y = apply_target_transform(y_raw, self.target_transform)
        y = self._standardize_target(y)

        sd = torch.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)
        sd = torch.where(valid_mask > 0.5, sd, torch.zeros_like(sd))

        return {
            "image": x.float(),
            "target": y.float(),
            "target_raw": y_raw.float(),
            "sd": sd.float(),
            "valid_mask": valid_mask.float(),
            "meta": meta.float(),
            "tile_id": torch.tensor(tile_id, dtype=torch.long),
        }


def make_datasets(cache_dir: str | Path, split_mode: str = "spatial", train_fraction: float = 0.8, val_fraction: float = 0.1, seed: int = 42, spatial_axis: str = "lon", subsample_fraction: float = 1.0, target_transform: str = "log1p", standardize_target: bool = True, normalize_input: bool = True, spatial_grid_rows: int = 4, spatial_grid_cols: int = 4) -> Tuple[BiomassTileDataset, BiomassTileDataset, BiomassTileDataset, Dict[str, object]]:
    """
    Convenience function used by scripts/train_eval.py.

    It:
        1. lists tile ids
        2. creates train/val/test split
        3. computes train-only target stats
        4. builds train/val/test Dataset objects
    """
    cache_dir = Path(cache_dir)
    tile_ids = list_tile_ids(cache_dir)

    train_ids, val_ids, test_ids, split_info = split_tile_ids(
        tile_ids=tile_ids,
        root_dir=cache_dir,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        split_mode=split_mode,
        seed=seed,
        spatial_axis=spatial_axis,
        spatial_grid_rows=spatial_grid_rows,
        spatial_grid_cols=spatial_grid_cols,
    )

    target_stats = compute_target_stats(
        tile_ids=train_ids,
        cache_dir=cache_dir,
        target_transform=target_transform,
    )

    train_ds = BiomassTileDataset(
        tile_ids=train_ids,
        cache_dir=cache_dir,
        target_stats=target_stats,
        target_transform=target_transform,
        standardize_target=standardize_target,
        normalize_input=normalize_input,
    )

    val_ds = BiomassTileDataset(
        tile_ids=val_ids,
        cache_dir=cache_dir,
        target_stats=target_stats,
        target_transform=target_transform,
        standardize_target=standardize_target,
        normalize_input=normalize_input,
    )

    test_ds = BiomassTileDataset(
        tile_ids=test_ids,
        cache_dir=cache_dir,
        target_stats=target_stats,
        target_transform=target_transform,
        standardize_target=standardize_target,
        normalize_input=normalize_input,
    )

    split_info = {
        "cache_dir": str(cache_dir),
        "num_tiles": len(tile_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "split_mode": split_mode,
        "spatial_axis": spatial_axis,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "subsample_fraction": subsample_fraction,
        "target_stats": target_stats,
        "target_transform": target_transform,
        "standardize_target": standardize_target,
        "normalize_input": normalize_input,
        "input_bands": INPUT_BANDS,
        "target_bands": TARGET_BANDS,
    }

    return train_ds, val_ds, test_ds, split_info