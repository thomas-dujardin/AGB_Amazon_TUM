#!/usr/bin/env bash
set -e

# Inference on one tile.
# Replace MODEL_CHECKPOINT with the best.pt path from a completed training run.

MODEL_CHECKPOINT="runs/baseline_frozen/YYYYMMDD_HHMMSS/checkpoints/best.pt"
TILE_ID=368

biomass-infer \
  --hf_dataset tdujardin/amazon_basin_pt_320 \
  --hf_subdir biomass_tiles_320_pt \
  --hf_local_dir data/hf \
  --model_checkpoint "${MODEL_CHECKPOINT}" \
  --tile_id "${TILE_ID}" \
  --output_dir "inference_outputs/tile_${TILE_ID}" \
  --save_pred_pt \
  --save_png \
  --save_csv \
  --batch_size 1 \
  --num_workers 0 \
  --use_gpu

# Hyperparameter notes:
# MODEL_CHECKPOINT: path to runs/<experiment>/<timestamp>/checkpoints/best.pt.
# TILE_ID: tile index to run inference on.
# --hf_dataset: HF dataset repo containing the tile tensors.
# --hf_subdir: subfolder containing tile_XXXXXX folders.
# --hf_local_dir: local cache for HF files.
# --model_checkpoint: trained biomass model checkpoint.
# --tile_id: run one tile only.
# --output_dir: where predictions, PNG img and CSV are saved.
# --save_pred_pt: save raw 32x32 AGB prediction tensor.
# --save_png: save target/pred/error/SD/mask viz.
# --save_csv: save per-tile metrics.
# --batch_size: set to 1.