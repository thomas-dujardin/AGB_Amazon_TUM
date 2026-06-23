#!/usr/bin/env bash
set -e

python scripts/train_eval.py \
  --experiment_name refiner \
  --hf_dataset tdujardin/amazon_basin_pt_320 \
  --hf_subdir biomass_tiles_320_pt \
  --hf_local_dir data/hf \
  --checkpoint_path checkpoints/CopernicusFM_ViT_base_varlang_e100.pth \
  --no-train_encoder \
  --refiner_on \
  --refiner_width 64 \
  --refiner_dropout 0.0 \
  --loss_type l1 \
  --target_transform log1p \
  --standardize_target \
  --normalize_input \
  --split_mode spatial \
  --spatial_axis lon \
  --epochs 30 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --lr 3e-4 \
  --num_workers 4