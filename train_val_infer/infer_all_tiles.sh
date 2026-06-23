#!/usr/bin/env bash
set -e

# Inference on all tiles.
# Replace MODEL_CHECKPOINT with the best.pt path from a completed training run.

MODEL_CHECKPOINT="runs/baseline_frozen/YYYYMMDD_HHMMSS/checkpoints/best.pt"

biomass-infer \
  --hf_dataset tdujardin/amazon_basin_pt_320 \
  --hf_subdir biomass_tiles_320_pt \
  --hf_local_dir data/hf \
  --model_checkpoint "${MODEL_CHECKPOINT}" \
  --all_tiles \
  --output_dir inference_outputs/all_tiles \
  --save_pred_pt \
  --save_png \
  --save_csv \
  --batch_size 4 \
  --num_workers 4 \
  --use_gpu

# Hyperparameter notes:
# MODEL_CHECKPOINT: path to runs/<experiment>/<timestamp>/checkpoints/best.pt.
# --all_tiles: run inference over every complete tile folder.
# --output_dir: where all prediction folders and summary CSV/JSON are saved.
# --save_pred_pt: save each raw 32x32 AGB prediction tensor.
# --save_png: save target/pred/error/SD/mask for each tile.
# --save_csv: save per-tile summary.
# --batch_size: inference batch size.