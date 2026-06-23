#!/usr/bin/env bash
set -e

python scripts/train_eval.py \
  --experiment_name quick_debug \
  --hf_dataset tdujardin/amazon_basin_pt_320 \
  --hf_subdir biomass_tiles_320_pt \
  --hf_local_dir data/hf \
  --checkpoint_path checkpoints/CopernicusFM_ViT_base_varlang_e100.pth \
  --no-train_encoder \
  --no-refiner_on \
  --loss_type l1 \
  --target_transform log1p \
  --standardize_target \
  --normalize_input \
  --split_mode random \
  --subsample_fraction 0.1 \
  --epochs 2 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --num_workers 0 \
  --log_every 1 \
  --check_forward