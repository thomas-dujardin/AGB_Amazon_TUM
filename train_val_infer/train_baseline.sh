#!/usr/bin/env bash
set -e

# Frozen Copernicus-FM baseline.
# Outputs: runs/baseline_frozen/<timestamp>/
# Validation is run every epoch; the best validation checkpoint is evaluated on the held-out test split.

biomass-train \
  --experiment_name baseline_frozen \
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
  --monitor val_rmse \
  --check_forward

# Hyperparameter notes:
# --experiment_name: run name under runs/.
# --hf_dataset: HF dataset repo containing the tensor tiles.
# --hf_subdir: subfolder containing tile_XXXXXX folders.
# --hf_local_dir: local cache for HF files.
# --checkpoint_path: pretrained Copernicus-FM checkpoint.
# --no-train_encoder: freeze Copernicus-FM.
# --no-refiner_on: disable optional 32x32 residual refiner.
# --loss_type l1: masked L1 loss on 32x32 transformed AGB.
# --laplacian_coeff: optional 32x32 structure loss coefficient.
# --target_transform log1p: train on log(1 + AGB).
# --standardize_target: normalize transformed target with train mean/std.
# --normalize_input: per-tile input channel normalization.
# --split_mode spatial: spatial split rather than random tile split.
# --spatial_axis lon: sort tiles by longitude for spatial split.
# --train_fraction / --val_fraction: remaining data becomes test split.
# --subsample_fraction: 1.0 means use all tiles.
# --epochs: number of training epochs.
# --batch_size: training batch size.
# --eval_batch_size: validation/test batch size.
# --lr: decoder/refiner learning rate.
# --encoder_lr: encoder learning rate if encoder is unfrozen.
# --weight_decay: AdamW weight decay.
# --eta_min: cosine scheduler minimum LR.
# --grad_clip: max gradient norm.
# --num_workers: DataLoader workers.
# --log_every: TensorBoard scalar logging frequency.
# --eval_every: validation frequency in epochs.
# --save_viz_every: TensorBoard image logging frequency in epochs.
# --num_viz: number of tiles visualized.
# --monitor: metric used to select best.pt.
# --check_forward: sanity-check model dimensions before training.