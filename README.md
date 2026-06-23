# <p align="center">Use Case 2b, Work Packages 4 and 5</p>
### <p align="center">Project: 101130544 — ThinkingEarth — HORIZON-EUSPA-2022-SPACE</p>

# <h1 align="center">Biomass Density Estimation in the Amazon basin using Synthetic Aperture Radar (SAR) and multispectral (MS) data</h1>

[![License: Code](https://img.shields.io/badge/License--Code-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![License: model](https://img.shields.io/badge/License--Model-CC--BY--4.0-blue.svg)]([https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/))

## Overview

This repository contains the implementation of biomass density estimation for the Amazon basin in **Use Case 2b**.

As the Grant Agreement stipulates:

> <p align="center">For UC2b, we will estimate AGB and assess forest carbon stock at fine spatial resolution and large geographic coverage using DL. To achieve this, we will utilize GEDI LiDAR data fused with S1&S2, elevation, meteo and land cover data for the Amazon basin. We will also model the amount of sequestered carbon and its year-to-year dynamics, as well as the performance of forests in carbon sinks for any location, country, or specific carbon REDD+ project areas. All DL models will incorporate xAI, and we will thoroughly evaluate the generalization of methods.</p>

## Current study area

For out-of-distribution generalization studies, as well as initial performance and speed testing, the current model is trained on a small area near Manaus rather than on the entire Amazon basin.

![Study area near Manaus](https://github.com/thomas-dujardin/biomass_density_ESA/blob/main/assets/tiles.png?raw=true)

The current dataset contains **384 complete tiles**. Each tile covers approximately `3.2 km × 3.2 km`.

This repository currently estimates biomass density only within the area shown above. The study area must later be extended and diversified to evaluate generalization across the Amazon basin and support the remaining objectives of Use Case 2b.

## Current approach

The current implementation uses a slightly modified version of the pretrained [Copernicus-FM v1](https://github.com/zhu-xlab/Copernicus-FM) multimodal foundation model, followed by task-specific regression components.

The model predicts ESA CCI Above-Ground Biomass maps at the target resolution.

![Biomass density estimation approach](https://github.com/thomas-dujardin/biomass_density_ESA/blob/main/assets/simplified_version_improved.png?raw=true)

Copernicus-FM v1 is used because it can jointly process heterogeneous Sentinel-1 SAR and Sentinel-2 optical observations.

The current approach does **not** include GEDI data because Copernicus-FM v1 does not natively support LiDAR inputs. GEDI integration therefore remains outside the present implementation.

## Input data

Each input tile is stored as a tensor with shape:

```text
8 × 320 × 320
```

where the dimensions correspond to `(channels, height, width)`.

The inputs are represented on a common `10 m` grid and contain:

| Source             | Bands                                    |
| ------------------ | ---------------------------------------- |
| Sentinel-1 SAR     | `VV`, `VH`                               |
| Sentinel-2 optical | `B02`, `B03`, `B04`, `B08`, `B11`, `B12` |

`B11` and `B12` are native `20 m` Sentinel-2 SWIR bands resampled onto the common `10 m` grid.

## Target data

The target data is derived from ESA CCI Above-Ground Biomass and is stored as three separate tensors, each with shape:

```text
1 × 32 × 32
```

| Tensor       | Meaning                                             |
| ------------ | --------------------------------------------------- |
| `AGB`        | Above-Ground Biomass density in `Mg/ha`             |
| `SD`         | ESA CCI standard deviation and uncertainty estimate |
| `valid_mask` | Binary mask indicating valid AGB target cells       |

The model predicts only the `AGB` tensor.

`valid_mask` is used to exclude invalid target cells from losses and evaluation metrics. `SD` is not passed to the model, but it can optionally be used to weight the loss according to ESA CCI target uncertainty.


## Target data

## Data representation and model architecture

The task is **regression**, not semantic segmentation: the model predicts continuous ESA CCI Above-Ground Biomass density values in **Mg/ha**.

Each tile covers approximately `3.2 km × 3.2 km` and contains:

| Tensor       |  Shape per tile | Role                                                   |
| ------------ | --------------: | ------------------------------------------------------ |
| `x`          | `8 × 320 × 320` | Sentinel-1 and Sentinel-2 inputs on a common 10 m grid |
| `AGB`        |   `1 × 32 × 32` | Biomass regression target at 100 m resolution          |
| `SD`         |   `1 × 32 × 32` | ESA CCI uncertainty estimate                           |
| `valid_mask` |   `1 × 32 × 32` | Indicates which AGB cells are valid                    |

For a batch of size `B`, the leading dimension becomes `B`.

The eight input channels are:

```text
VV, VH, B02, B03, B04, B08, B11, B12
```

`VV` and `VH` come from Sentinel-1 SAR. The remaining bands come from Sentinel-2. `B11` and `B12` are native 20 m SWIR bands resampled onto the common 10 m input grid.

### Why use 320 × 320 inputs?

The chosen size aligns both with the Copernicus-FM backbone and with the ESA target grid:

```text
320 / 16 = 20   → regular 20 × 20 ViT patch grid
320 / 10 = 32   → 32 target cells across the tile
```

Copernicus-FM v1 uses a ViT-B/16 backbone, while the input and target resolutions are respectively 10 m and 100 m. Therefore, a `320 × 320` input naturally corresponds to a `32 × 32` AGB target.

### Forward pass

```text
Input
(B, 8, 320, 320)

        ↓ Copernicus-FM v1, ViT-B/16

Patch tokens
(B, 400, 768)
400 = 20 × 20 patches
768 = embedding dimension

        ↓ reshape patch sequence into a spatial grid

Feature map
(B, 768, 20, 20)

        ↓ Conv/GELU
(B, 256, 20, 20)

        ↓ resize to target grid
(B, 256, 32, 32)

        ↓ Conv/GELU
(B, 128, 32, 32)

        ↓ Conv/GELU
(B, 64, 32, 32)

        ↓ prediction head

Coarse AGB prediction
(B, 1, 32, 32)
```

The modified Copernicus-FM backbone returns all `400` spatial patch tokens after removing the CLS token.

An optional lightweight residual CNN can refine the coarse prediction directly at target resolution:

```text
Coarse prediction
(B, 1, 32, 32)

        ↓ optional residual CNN refiner

Final prediction
(B, 1, 32, 32)
```

If the refiner is disabled, the coarse prediction is used directly.

### Targets, losses, and metrics

The model predicts only AGB. The loss is computed against the AGB target and restricted to valid cells using `valid_mask`.

`SD` is not passed to the model. It can optionally weight the loss so that uncertain ESA CCI cells contribute less.

Because AGB values are strongly skewed, training can use the transformation:

```text
y = log(1 + AGB)
```

The transformed values may also be standardized using the training-set mean and standard deviation. This reduces the influence of very high biomass values and generally makes optimization more stable.

Before reporting metrics, the transformation is reversed:

```text
model output
→ undo standardization
→ exp(y) - 1
→ raw AGB prediction in Mg/ha
```

MAE, RMSE, and bias are then computed on the raw `32 × 32` AGB maps, using only cells selected by `valid_mask`.


## Model Overview

![biomass density estimation diagram](https://github.com/thomas-dujardin/biomass_density_ESA/blob/main/assets/biomass_redone.png?raw=true)

<p align="left"> The original Copernicus-FM ViT uses a `[CLS]` token for global image representation. In this repository, `src/model_vit.py` is modified so that the model returns the spatial patch tokens instead. These tokens are reshaped into a 20 × 20 features grid before being decoded into an AGB map.</p>

## Installation

run (...)

conda create -n biomass python=3.11
conda activate biomass
pip install -r requirements.txt

## Minimal command, experiment examples, inference
- ### Runs the model with the random seed and hyperparameters we've used, on the same data split. The Copernicus-FM v1 encoder is frozen, and the refiner is turned off;
  python biomass_density.py --use_gpu --nr_epochs 20 --train_batch_size 4

- ### Frozen Copernicus-FM encoder, train decoder/refiner
  python biomass_density.py
  
- ### Train all components
  python biomass_density.py --train_everything

- ### Randomly initialize Copernicus-FM
  python biomass_density.py --random_init_copernicus

- ### Disable refiner
  python biomass_density.py --no-refiner_on

These arguments can be combined.

## Inference

Run:

python infer_biomass.py \
  --input_tensor data/cache/tile_000123/x.pt \
  --biomass_checkpoint checkpoints/biomass_best.pt \
  --out_dir data/predictions/tile_000123

## Experimental results:

## Results

Some shell scripts are present in the **/experiments** folder. They launch training or inference with various hyperparameters.

**/train_val_infer** contain the best set of hyperparameters for training and inference. They should be used for inference on unseen data, as they use the best trained model available.

# log1p transform ablation

5 runs on 5 different seeds, ran with **experiments/repeated_spatial_blocks_for_UQ.sh**. No refiner, frozen CFM-v1 backbone. **Uses log1p transform**:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.28 ± 3.75 Mg/ha |
| RMSE   | 74.73 ± 4.30 Mg/ha |
| Bias   |  1.56 ± 8.94 Mg/ha |

5 runs on 5 different seeds, no log1p transform to the AGB, frozen CFM-v1 and no refiner:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.56 ± 3.61 Mg/ha |
| RMSE   | 75.16 ± 3.99 Mg/ha |
| Bias   |  1.55 ± 9.73 Mg/ha |

# CFM-v1 initialization ablation study

5 runs on 5 different seeds, log1p on, pretrained CFM-v1 finetuned, refiner activated ("maximal" config):

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 53.89 ± 3.60 Mg/ha |
| RMSE   | 67.21 ± 4.12 Mg/ha |
| Bias   | −2.04 ± 8.56 Mg/ha |

5 runs on 5 different seeds, CFM-v1 randomly initialized but finetuned, "maximal" config on (refiner, log1p, training the randomly initialized CFM-v1):

| Metric |           Mean ± SD |
| ------ | ------------------: |
| MAE    |  53.61 ± 2.75 Mg/ha |
| RMSE   |  66.76 ± 3.32 Mg/ha |
| Bias   | −1.39 ± 12.47 Mg/ha |

# CFM-v1 with SD-based losses ablation study

5 runs on 5 different seeds, CFM-v1 frozen, no refiner, log1p and **SD_weighted_L1_loss instead of L1**:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.49 ± 2.86 Mg/ha |
| RMSE   | 75.02 ± 3.11 Mg/ha |
| Bias   |  2.72 ± 9.32 Mg/ha |

5 runs on 5 different seeds, same parameters as above, but **simple unweighted L1 loss**:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.47 ± 3.26 Mg/ha |
| RMSE   | 75.06 ± 3.53 Mg/ha |
| Bias   |  2.44 ± 9.69 Mg/ha |

5 runs on 5 different seeds, same parameters as above, but **SD_weighted_MSE_loss**:

| Metric |            Mean ± SD |
| ------ | -------------------: |
| MAE    |   59.94 ± 4.16 Mg/ha |
| RMSE   |   74.78 ± 4.89 Mg/ha |
| Bias   | −12.56 ± 12.13 Mg/ha |

5 runs on 5 different seeds, same parameters as above, but **unweighted_MSE_loss**:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.15 ± 4.74 Mg/ha |
| RMSE   | 73.77 ± 5.38 Mg/ha |
| Bias   | −9.41 ± 8.75 Mg/ha |

5 runs on 5 different seeds, same parameters as above, but **Laplacian penalty added to L1 loss**:



5 runs on 5 different seeds, same parameters as above, but **simple L1 loss**:

# CFM-v1 with and without refiner ablation study

5 runs on 5 different seeds, CFM-v1 frozen, **refiner activated**, L1 loss, log1p:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 58.83 ± 3.88 Mg/ha |
| RMSE   | 74.14 ± 4.13 Mg/ha |
| Bias   | 3.12 ± 13.33 Mg/ha |

5 runs on 5 different seeds, CFM-v1 frozen, **refiner removed**, L1 loss, log1p:

| Metric |          Mean ± SD |
| ------ | -----------------: |
| MAE    | 59.54 ± 3.69 Mg/ha |
| RMSE   | 74.97 ± 4.06 Mg/ha |
| Bias   | 1.27 ± 10.13 Mg/ha |

# Some qualitative results



| Setting | Encoder | Refiner | Loss | RMSE | MAE | Notes |
|---|---|---|---|---:|---:|---|
| baseline | frozen Copernicus-FM | off | L1 | TBD | TBD | coarse target |
| + refiner | frozen Copernicus-FM | UResNet34 | L1 + Laplacian | TBD | TBD | slower |
| random encoder | random ViT-B | off | L1 | TBD | TBD | ablation |

## Current limitations

- GEDI LiDAR is not currently used as an input modality. Could be implemented from scratch by correctly fusing the embeddings and the GEDI LiDAR data.
- The target biomass map is not a true 10 m pixelwise label; it has coarser effective spatial support.
- The current implementation requires access to private Google Earth Engine assets.
- The current model is trained on a small number of Amazon Basin tiles.
- Edge artifacts in GT tiles.
- Currently, single-GPU only.
