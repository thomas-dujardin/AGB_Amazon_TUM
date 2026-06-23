#!/usr/bin/env bash
set -e

# Fine-tunes Copernicus-FM with a small encoder learning rate.
# This is more expensive and riskier than the frozen baseline.

biomass-train \
  --experiment_name finetune_encoder \
  --hf_dataset tdujardin/amazon_basin_pt_320 \
  --hf_subdir biomass_tiles_320_pt \
  --hf_local_dir data/hf \
  --checkpoint_path checkpoints/CopernicusFM_ViT_base_varlang_e100.pth \
  --train_encoder \
  --no-refiner_on \
  --loss_type l1 \
  --laplacian_coeff 0.0 \
  --target_transform log1p \
  --standardize_target \
  --normalize_input \
  --split_mode spatial \
  --spatial_axis lon \
  --train_fraction 0.8 \
  --val_fraction 0.1 \
  --subsample_fraction 1.0 \
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

# Hyperparameter notes:
# --train_encoder: unfreezes Copernicus-FM.
# --encoder_lr: small learning rate for encoder parameters.
# --lr: learning rate for decoder/refiner parameters.
# All other args have the same meaning as in train_baseline.sh.