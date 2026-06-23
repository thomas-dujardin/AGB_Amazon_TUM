#!/usr/bin/env bash
set -e

# Repeated spatial block validation/test protocol.
#
# This launches 5 independent train/val/test runs.
# Each run:
#   1. divides the AOI into spatial blocks,
#   2. assigns whole blocks to train/val/test,
#   3. trains a model,
#   4. reloads the best validation checkpoint,
#   5. evaluates on the held-out test blocks.
#
# The goal is not to get one lucky split, but to estimate:
#   mean ± std test RMSE / MAE / bias across several spatial block splits.

SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo "============================================================"
  echo "Running spatial block split seed ${SEED}"
  echo "============================================================"
  echo ""

  biomass-train \
    --experiment_name "spatial_blocks_seed_${SEED}" \
    --hf_dataset tdujardin/amazon_basin_pt_320 \
    --hf_subdir biomass_tiles_320_pt \
    --hf_local_dir data/hf \
    --checkpoint_path checkpoints/CopernicusFM_ViT_base_varlang_e100.pth \
    --no-train_encoder \
    --no-refiner_on \
    --loss_type l1 \
    --laplacian_coeff 0.0 \
    --target_transform log1p \
    --standardize_target \
    --normalize_input \
    --split_mode spatial_blocks \
    --spatial_grid_rows 4 \
    --spatial_grid_cols 4 \
    --train_fraction 0.75 \
    --val_fraction 0.125 \
    --subsample_fraction 1.0 \
    --seed "${SEED}" \
    --epochs 30 \
    --batch_size 4 \
    --eval_batch_size 4 \
    --lr 3e-4 \
    --encoder_lr 1e-5 \
    --weight_decay 1e-4 \
    --eta_min 1e-6 \
    --grad_clip 1.0 \
    --num_workers 4 \
    --log_every 20 \
    --eval_every 1 \
    --save_viz_every 1 \
    --num_viz 2 \
    --monitor val_rmse
done