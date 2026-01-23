from dataclasses import dataclass, field
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import default_weights_path, get_encoder_spec

logger = logging.getLogger(__name__)

@dataclass
class UnetConfig:
    in_channels: int = 3
    out_channels: int = 150
    features: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_pretrained_encoder: bool = False
    encoder_name: str | None = None
    encoder_weights_path: str | None = None


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, config: UnetConfig):
        super().__init__()
        self.uses_pretrained_encoder = False
        self.encoder = None

        if config.use_pretrained_encoder and config.encoder_name:
            self._init_pretrained_encoder(config)
        else:
            self._init_plain_unet(config)

    def _init_plain_unet(self, config: UnetConfig) -> None:
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = config.in_channels
        for feature in config.features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(config.features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(
            config.features[-1], config.features[-1] * 2
        )
        self.final_conv = nn.Conv2d(config.features[0], config.out_channels, kernel_size=1)

    def _init_pretrained_encoder(self, config: UnetConfig) -> None:
        if config.in_channels != 3:
            raise ValueError("Pretrained encoders expect 3-channel RGB inputs.")
        spec = get_encoder_spec(config.encoder_name)
        self.encoder = self._build_resnet_encoder(spec, config.encoder_weights_path)
        c1, c2, c3, c4, c5 = spec.channels
        self.up1 = UpBlock(c5, c4, c4)
        self.up2 = UpBlock(c4, c3, c3)
        self.up3 = UpBlock(c3, c2, c2)
        self.up4 = UpBlock(c2, c1, c1)
        self.final_up = nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(c1, config.out_channels, kernel_size=1)
        self.uses_pretrained_encoder = True

    def _build_resnet_encoder(self, spec, weights_path: str | None) -> nn.Module:
        import torchvision.models as tv_models

        builder = getattr(tv_models, spec.builder)
        encoder = builder(weights=None)
        resolved_path = default_weights_path(spec.name)
        if weights_path:
            resolved_path = Path(weights_path)
        if resolved_path.is_file():
            state_dict = torch.load(resolved_path, map_location="cpu")
            encoder.load_state_dict(state_dict)
        else:
            logger.warning(
                "Pretrained weights not found for %s at %s. Using random init.",
                spec.name,
                resolved_path,
            )
        encoder.fc = nn.Identity()
        encoder.avgpool = nn.Identity()
        return encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_pretrained_encoder:
            return self._forward_pretrained(x)
        return self._forward_plain(x)

    def _forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

    def _forward_pretrained(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x1 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x2 = self.encoder.layer1(self.encoder.maxpool(x1))
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return self.final_conv(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(UnetConfig()).to(device)
    x = torch.randn((2, 3, 512, 512), device=device)
    with torch.no_grad():
        preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
