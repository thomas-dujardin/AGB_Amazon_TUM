from math import sqrt
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

INPUT_BANDS = ["VV", "VH", "B02", "B03", "B04", "B08", "B11", "B12"]

# The two SAR channels use artificial wavelength markers
# B11/B12 are native 20 m Sentinel-2 bands, but in the tensor they are resampled
# onto the common 10 m grid before being fed to Copernicus-FM.

DEFAULT_WAVELENGTHS = [
    50_000_000.0,  # VV
    50_000_001.0,  # VH
    490.0,         # B02
    560.0,         # B03
    665.0,         # B04
    842.0,         # B08
    1610.0,        # B11
    2190.0,        # B12
]

DEFAULT_BANDWIDTHS = [
    1e9,    # VV
    1e9,    # VH
    65.0,   # B02
    35.0,   # B03
    30.0,   # B04
    115.0,  # B08
    90.0,   # B11
    180.0,  # B12
]

def set_trainable(module: nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable

def init_vit_random_weights(
    model: nn.Module,
    token_std: float = 0.02,
    head_std: float = 1e-3,
) -> nn.Module:
    """
    Used only for the random-init Copernicus-FM ablation.
    """
    def is_special_parameter(name: str) -> bool:
        name = name.lower()
        return any(
            key in name
            for key in [
                "pos_embed",
                "position_embedding",
                "cls_token",
                "mask_token",
                "query_token",
                "register_token",
            ]
        )

    def is_head(name: str) -> bool:
        name = name.lower()
        return any(
            key in name
            for key in ["head", "classifier", "prediction", "lm_head"]
        )

    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if is_head(module_name):
                    nn.init.trunc_normal_(
                        module.weight,
                        mean=0.0,
                        std=head_std,
                        a=-2 * head_std,
                        b=2 * head_std,
                    )
                else:
                    nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if is_special_parameter(name):
                nn.init.trunc_normal_(
                    param,
                    mean=0.0,
                    std=token_std,
                    a=-2 * token_std,
                    b=2 * token_std,
                )

    return model


def load_copernicus_encoder(
    checkpoint_path: Optional[str],
    random_init: bool = False,
    train_encoder: bool = False,
) -> nn.Module:
    """
    Loads Copernicus-FM ViT-B/16 from src/.
    src/model_vit.py already returns:
        outcome = x[:, 1:, :]

        So the encoder output is expected to be patch tokens only, with no CLS token.

    For a 320x320 input:
        320 / 16 = 20
        20 * 20 = 400
    """
    from src.model_vit import vit_base_patch16

    encoder = vit_base_patch16()

    if random_init:
        encoder = init_vit_random_weights(encoder)

        checkpoint_path = Path(checkpoint_path)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        encoder.load_state_dict(state_dict, strict=False)

    set_trainable(encoder, train_encoder)
    return encoder


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    """
    Converts Copernicus-FM patch tokens to a spatial feature map.

    Expected input:
        tokens: [B, 400, 768]

    Output:
        grid: [B, 768, 20, 20]
    """

    b, n, c = tokens.shape
    h = int(sqrt(n))

    return tokens.transpose(1, 2).reshape(b, c, h, h)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        group_count = min(groups, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder20To32(nn.Module):
    """
    Maps the Copernicus-FM 20x20 token grid to the ESA CCI AGB 32x32 grid.

    Input:
        tokens: [B, 400, 768]

    Internal:
        [B, 400, 768]
        -> [B, 768, 20, 20]
        -> [B, hidden_dim, 20, 20]
        -> [B, hidden_dim/2, 20, 20]
        -> resize to [B, hidden_dim/2, 32, 32]

    Output:
        [B, out_channels, 32, 32]

    Usually:
        out_channels = 1
        output = normalized transformed AGB prediction
    """
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        out_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            ConvBlock(embed_dim, hidden_dim, dropout=dropout),
            ConvBlock(hidden_dim, hidden_dim // 2, dropout=dropout),
        )

        self.to_target_grid = nn.Upsample(
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )

        self.head = nn.Sequential(
            ConvBlock(hidden_dim // 2, 64, dropout=dropout),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens_to_grid(tokens)
        x = self.proj(x)
        x = self.to_target_grid(x)
        return self.head(x)


class SmallCoarseRefiner(nn.Module):
    """
    Small randomly initialized residual refiner at 32x32.

    It refines the target-resolution prediction, not the original 320x320 image.

    Input:
        coarse: [B, C, 32, 32]

    Output:
        refined: [B, C, 32, 32]

    Formula:
        refined = coarse + delta(coarse)
    """
    def __init__(
        self,
        channels: int = 1,
        width: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.enc1 = ConvBlock(channels, width, dropout=dropout)

        self.down = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.enc2 = ConvBlock(width * 2, width * 2, dropout=dropout)
        self.mid = ConvBlock(width * 2, width * 2, dropout=dropout)

        self.up = nn.Sequential(
            nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.dec = ConvBlock(width * 2, width, dropout=dropout)
        self.out = nn.Conv2d(width, channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(coarse)

        x2 = self.down(x1)
        x2 = self.enc2(x2)
        x2 = self.mid(x2)

        u = self.up(x2)
        u = torch.cat([u, x1], dim=1)

        delta = self.out(self.dec(u))
        return coarse + delta

class CopernicusBiomassModel(nn.Module):
    """
    Full biomass model.

    Dimension flow:
        x:
            [B, 8, 320, 320]

        Copernicus-FM ViT-B/16:
            [B, 8, 320, 320] -> [B, 400, 768]

        Decoder20To32:
            [B, 400, 768] -> [B, 1, 32, 32]

        Optional refiner:
            [B, 1, 32, 32] -> [B, 1, 32, 32]

    SD is not passed into the model. It is used only in the loss or metrics.
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        refiner: Optional[nn.Module] = None,
        wavelengths: Optional[Iterable[float]] = None,
        bandwidths: Optional[Iterable[float]] = None,
        input_mode: str = "spectral",
        kernel_size: int = 16,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.refiner = refiner

        self.wavelengths = (
            list(wavelengths)
            if wavelengths is not None
            else list(DEFAULT_WAVELENGTHS)
        )
        self.bandwidths = (
            list(bandwidths)
            if bandwidths is not None
            else list(DEFAULT_BANDWIDTHS)
        )

        self.input_mode = input_mode
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        _, tokens = self.encoder(
            x,
            meta,
            self.wavelengths,
            self.bandwidths,
            None,
            self.input_mode,
            self.kernel_size,
        )

        pred = self.decoder(tokens)

        if self.refiner is not None:
            pred = self.refiner(pred)

        return pred


def build_model(
    checkpoint_path: Optional[str],
    random_init_copernicus: bool = False,
    train_encoder: bool = False,
    decoder_hidden_dim: int = 256,
    decoder_dropout: float = 0.0,
    refiner_on: bool = False,
    refiner_width: int = 64,
    refiner_dropout: float = 0.0,
    out_channels: int = 1,
) -> CopernicusBiomassModel:
    """
    Builder used by scripts/train_eval.py.
    """
    encoder = load_copernicus_encoder(
        checkpoint_path=checkpoint_path,
        random_init=random_init_copernicus,
        train_encoder=train_encoder,
    )

    decoder = Decoder20To32(
        embed_dim=768,
        hidden_dim=decoder_hidden_dim,
        out_channels=out_channels,
        dropout=decoder_dropout,
    )

    refiner = None
    if refiner_on:
        refiner = SmallCoarseRefiner(
            channels=out_channels,
            width=refiner_width,
            dropout=refiner_dropout,
        )

    return CopernicusBiomassModel(
        encoder=encoder,
        decoder=decoder,
        refiner=refiner,
    )


@torch.no_grad()
def check_forward_320(
    model: CopernicusBiomassModel,
    device: torch.device,
    batch_size: int = 2,
) -> torch.Tensor:
    """
    Expected:
        output.shape = [batch_size, 1, 32, 32]
    """
    model.eval()

    x = torch.randn(batch_size, 8, 320, 320, device=device)
    meta = torch.zeros(batch_size, 4, device=device)

    return model(x, meta)