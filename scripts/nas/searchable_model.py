from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from scripts.nas.search_space import ArchitectureConfig


CHANNEL_TO_INDEX: Dict[str, int] = {
    "AWGN": 0,
    "Fading": 1,
    "Combined_channel": 2,
}


def _channel_index_tensor(channel_type: str, batch_size: int, device: torch.device) -> torch.Tensor:
    idx = CHANNEL_TO_INDEX.get(channel_type)
    if idx is None:
        raise ValueError(f"Unsupported channel_type: {channel_type}")
    return torch.full((batch_size,), idx, dtype=torch.long, device=device)


def _safe_cr_tensor(cr: Union[float, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(cr, torch.Tensor):
        if cr.dim() == 0:
            cr = cr.expand(batch_size)
        if cr.shape[0] != batch_size:
            raise ValueError(f"cr tensor batch mismatch: got {cr.shape[0]}, expected {batch_size}")
        return torch.clamp(cr.to(device=device, dtype=torch.float32), 1e-3, 1.0)
    return torch.full((batch_size,), float(max(1e-3, min(1.0, cr))), device=device, dtype=torch.float32)


def _make_topk_mask(weights: torch.Tensor, cr: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Build a hard channel mask by keeping top-k channels for each sample.
    weights: [B, C], cr can be scalar or per-sample tensor in (0, 1].
    """
    batch_size, channels = weights.shape
    cr_tensor = _safe_cr_tensor(cr, batch_size=batch_size, device=weights.device)
    k_tensor = torch.clamp(torch.round(cr_tensor * channels).long(), min=1, max=channels)

    mask = torch.zeros_like(weights)
    for i in range(batch_size):
        k = int(k_tensor[i].item())
        _, indices = torch.topk(weights[i], k=k, dim=0, largest=True, sorted=False)
        mask[i, indices] = 1.0
    return mask


def awgn_channel(x: torch.Tensor, snr_db: float, power: float = 2.0) -> torch.Tensor:
    gamma = 10 ** (snr_db / 10.0)
    std = torch.sqrt(torch.tensor(power / gamma, device=x.device, dtype=x.dtype))
    noise = std * torch.randn_like(x)
    return x + noise


def fading_channel(x: torch.Tensor, snr_db: float, power: float = 2.0) -> torch.Tensor:
    """
    Flat Rayleigh fading + additive Gaussian noise, then ideal equalization.
    Input shape: [B, F] (real-valued vector).
    """
    gamma = 10 ** (snr_db / 10.0)
    b, feature_len = x.shape
    need_pad = feature_len % 2 == 1

    if need_pad:
        x = torch.cat([x, torch.zeros((b, 1), device=x.device, dtype=x.dtype)], dim=1)
        feature_len = x.shape[1]

    k = feature_len // 2
    x_complex = torch.complex(x[:, 0::2], x[:, 1::2])
    h_i = torch.randn((b, k), device=x.device, dtype=x.dtype)
    h_r = torch.randn((b, k), device=x.device, dtype=x.dtype)
    h = torch.complex(h_i, h_r)

    y = h * x_complex
    std = torch.sqrt(torch.tensor(power / gamma, device=x.device, dtype=x.dtype))
    n_i = std * torch.randn((b, k), device=x.device, dtype=x.dtype)
    n_r = std * torch.randn((b, k), device=x.device, dtype=x.dtype)
    noise = torch.complex(n_i, n_r)
    y = (y + noise) / h

    y_out = torch.zeros((b, feature_len), device=x.device, dtype=x.dtype)
    y_out[:, 0::2] = y.real
    y_out[:, 1::2] = y.imag

    if need_pad:
        y_out = y_out[:, :-1]
    return y_out


def combined_channel(
    x_flat: torch.Tensor,
    snr_db: float,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    power: float = 2.0,
) -> torch.Tensor:
    # First apply Rayleigh fading on flattened symbols.
    x_faded = fading_channel(x_flat, snr_db, power=power).view(batch_size, channels, height, width)
    # Then apply AWGN with spatially varying SNR map.
    snr_map = torch.randint(
        low=0,
        high=29,
        size=(batch_size, channels, height, 1),
        device=x_flat.device,
    ).float()
    gamma = 10 ** (snr_map / 10.0)
    std = torch.sqrt(torch.tensor(power, device=x_flat.device, dtype=x_flat.dtype) / gamma)
    return x_faded + std * torch.randn_like(x_faded)


class ChannelConditionEncoder(nn.Module):
    """
    Encodes channel type and SNR into a compact conditioning vector.
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.channel_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, channel_type: str, snr_db: float, batch_size: int, device: torch.device) -> torch.Tensor:
        channel_idx = _channel_index_tensor(channel_type, batch_size=batch_size, device=device)
        channel_vec = self.channel_embed(channel_idx)

        snr_norm = float(snr_db) / 28.0
        snr_input = torch.full((batch_size, 1), snr_norm, device=device, dtype=torch.float32)
        snr_vec = self.snr_mlp(snr_input)
        return torch.cat([channel_vec, snr_vec], dim=1)


class ChannelConditionedSelector(nn.Module):
    """
    Channel gating network conditioned on semantic features + channel state.
    """

    def __init__(self, channels: int, ratio: int, cond_dim: int = 32):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.semantic_proj = nn.Linear(channels, hidden, bias=False)
        self.cond_proj = nn.Linear(cond_dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, semantic_vec: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        fused = self.semantic_proj(semantic_vec) + self.cond_proj(cond_vec)
        fused = self.act(fused)
        return self.gate(self.out_proj(fused))


class RateController(nn.Module):
    """
    Predicts sample-wise compression ratio (cr_i) from semantics and channel state.
    """

    def __init__(self, channels: int, cond_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels + cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        semantic_vec: torch.Tensor,
        cond_vec: torch.Tensor,
        base_cr: float,
        min_cr: float,
        max_cr: float,
        blend_alpha: float,
    ) -> torch.Tensor:
        raw = self.net(torch.cat([semantic_vec, cond_vec], dim=1)).squeeze(1)
        dynamic_cr = min_cr + (max_cr - min_cr) * raw
        base = torch.full_like(dynamic_cr, float(base_cr))
        blended = (1.0 - blend_alpha) * base + blend_alpha * dynamic_cr
        return torch.clamp(blended, min=min_cr, max=max_cr)


class SearchableSEBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ratio: int,
        use_channel_condition: bool = True,
        use_dynamic_rate: bool = True,
        min_dynamic_cr: float = 0.3,
        max_dynamic_cr: float = 1.0,
        rate_blend_alpha: float = 0.7,
    ):
        super().__init__()
        self.use_channel_condition = use_channel_condition
        self.use_dynamic_rate = use_dynamic_rate
        self.min_dynamic_cr = min_dynamic_cr
        self.max_dynamic_cr = max_dynamic_cr
        self.rate_blend_alpha = rate_blend_alpha

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fallback selector if channel-conditioned gating is disabled.
        hidden = max(1, channels // ratio)
        self.base_fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

        self.condition_encoder = ChannelConditionEncoder(embed_dim=16)
        self.cond_selector = ChannelConditionedSelector(channels=channels, ratio=ratio, cond_dim=32)
        self.rate_controller = RateController(channels=channels, cond_dim=32, hidden_dim=64)

    def forward(self, x: torch.Tensor, base_cr: float, channel_type: str, snr_db: float) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = x.shape
        semantic_vec = self.gap(x).view(b, c)
        cond_vec = self.condition_encoder(channel_type=channel_type, snr_db=snr_db, batch_size=b, device=x.device)

        if self.use_channel_condition:
            weights = self.cond_selector(semantic_vec=semantic_vec, cond_vec=cond_vec)
        else:
            weights = self.base_fc(semantic_vec)

        if self.use_dynamic_rate:
            cr_values = self.rate_controller(
                semantic_vec=semantic_vec,
                cond_vec=cond_vec,
                base_cr=base_cr,
                min_cr=self.min_dynamic_cr,
                max_cr=self.max_dynamic_cr,
                blend_alpha=self.rate_blend_alpha,
            )
        else:
            cr_values = torch.full((b,), float(base_cr), device=x.device, dtype=x.dtype)

        mask = _make_topk_mask(weights, cr=cr_values).view(b, c, 1, 1)
        return x * mask, cr_values


def _build_channel_schedule(in_ch: int, bottleneck_ch: int, depth: int) -> List[int]:
    # Build monotonic channels from in_ch to bottleneck_ch with "depth" transitions.
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if depth == 1:
        return [in_ch, bottleneck_ch]
    schedule = []
    for i in range(depth + 1):
        ratio = i / depth
        ch = int(round(in_ch * (1 - ratio) + bottleneck_ch * ratio))
        schedule.append(max(1, ch))
    schedule[0] = in_ch
    schedule[-1] = bottleneck_ch
    return schedule


class SearchableAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        depth: int,
        kernel_size: int,
        use_skip: bool,
        latent_noise_std: float = 0.1,
    ):
        super().__init__()
        if kernel_size not in (1, 3, 5):
            raise ValueError("kernel_size must be in {1, 3, 5}")
        padding = kernel_size // 2
        self.use_skip = use_skip
        self.latent_noise_std = latent_noise_std

        channel_schedule = _build_channel_schedule(in_channels, bottleneck_channels, depth=depth)
        encoder_layers: List[nn.Module] = []
        for in_ch, out_ch in zip(channel_schedule[:-1], channel_schedule[1:]):
            encoder_layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding),
                    nn.ReLU(inplace=True),
                ]
            )
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_schedule = list(reversed(channel_schedule))
        decoder_layers: List[nn.Module] = []
        for idx, (in_ch, out_ch) in enumerate(zip(decoder_schedule[:-1], decoder_schedule[1:])):
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            if idx < len(decoder_schedule) - 2:
                decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, channel_type: str, snr_db: float) -> torch.Tensor:
        residual = x
        z = self.encoder(x)
        if self.training and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std

        b, c, h, w = z.shape
        if channel_type == "AWGN":
            z_noisy = awgn_channel(z, snr_db)
        elif channel_type == "Fading":
            z_flat = z.reshape(b, -1)
            z_noisy = fading_channel(z_flat, snr_db).view(b, c, h, w)
        elif channel_type == "Combined_channel":
            z_flat = z.reshape(b, -1)
            z_noisy = combined_channel(
                z_flat,
                snr_db=snr_db,
                batch_size=b,
                channels=c,
                height=h,
                width=w,
            )
        else:
            raise ValueError(f"Unsupported channel_type: {channel_type}")

        out = self.decoder(z_noisy)
        if self.use_skip and out.shape == residual.shape:
            out = out + residual
        return out


class ChannelAwareClassifier(nn.Module):
    """
    ResNet18 backbone + searchable SE + searchable JSCC autoencoder.
    Insertion can happen after layer3 or after layer4.

    New in this version:
    - Channel-conditioned feature selector.
    - Dynamic sample-wise compression ratio predictor.
    """

    def __init__(
        self,
        num_classes: int,
        arch: ArchitectureConfig,
        pretrained_backbone: bool = True,
        use_channel_condition: bool = True,
        use_dynamic_rate: bool = True,
        min_dynamic_cr: float = 0.3,
        max_dynamic_cr: float = 1.0,
        rate_blend_alpha: float = 0.7,
    ):
        super().__init__()
        self.arch = arch
        self.last_forward_stats: Dict[str, float] = {
            "mean_cr": float(arch.cr),
            "std_cr": 0.0,
            "min_cr": float(arch.cr),
            "max_cr": float(arch.cr),
        }

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)

        feature_channels = 256 if arch.insertion_stage == 3 else 512
        self.se_block = SearchableSEBlock(
            channels=feature_channels,
            ratio=arch.se_ratio,
            use_channel_condition=use_channel_condition,
            use_dynamic_rate=use_dynamic_rate,
            min_dynamic_cr=min_dynamic_cr,
            max_dynamic_cr=max_dynamic_cr,
            rate_blend_alpha=rate_blend_alpha,
        )
        self.jscc_block = SearchableAutoencoder(
            in_channels=feature_channels,
            bottleneck_channels=arch.bottleneck_channels,
            depth=arch.ae_depth,
            kernel_size=arch.kernel_size,
            use_skip=arch.use_skip,
        )

    def _record_cr_stats(self, cr_values: torch.Tensor) -> None:
        cr_detached = cr_values.detach().float()
        self.last_forward_stats = {
            "mean_cr": float(cr_detached.mean().item()),
            "std_cr": float(cr_detached.std(unbiased=False).item()),
            "min_cr": float(cr_detached.min().item()),
            "max_cr": float(cr_detached.max().item()),
        }

    def forward(self, x: torch.Tensor, channel_type: str, snr_db: float) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.arch.insertion_stage == 3:
            x, cr_values = self.se_block(x, base_cr=self.arch.cr, channel_type=channel_type, snr_db=snr_db)
            self._record_cr_stats(cr_values)
            x = self.jscc_block(x, channel_type=channel_type, snr_db=snr_db)
            x = self.layer4(x)
        else:
            x = self.layer4(x)
            x, cr_values = self.se_block(x, base_cr=self.arch.cr, channel_type=channel_type, snr_db=snr_db)
            self._record_cr_stats(cr_values)
            x = self.jscc_block(x, channel_type=channel_type, snr_db=snr_db)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
