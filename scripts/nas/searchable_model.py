from __future__ import annotations

"""
可搜索的信道感知分类模型定义。

核心思想：
1. Backbone 采用 ResNet18。
2. 在 layer3 或 layer4 后插入“SE 通道选择 + JSCC 自编码传输块”。
3. SE 选择支持：
   - 信道状态条件输入（channel_type + snr）
   - 按样本动态预测压缩率 CR（sample-wise CR）
4. JSCC 块支持三类信道仿真：AWGN / Fading / Combined_channel。
"""

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
    """
    将字符串信道类型映射为 embedding 索引张量 [B]。

    例子：
    - channel_type="AWGN" -> idx=0
    - batch_size=4 -> [0,0,0,0]
    """
    idx = CHANNEL_TO_INDEX.get(channel_type)
    if idx is None:
        raise ValueError(f"Unsupported channel_type: {channel_type}")
    return torch.full((batch_size,), idx, dtype=torch.long, device=device)


def _safe_cr_tensor(cr: Union[float, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
    """
    统一处理 CR 输入，输出形状为 [B] 的浮点张量，且范围裁剪到 [1e-3, 1.0]。

    兼容两种输入：
    - 标量（float）：所有样本共享同一 CR
    - 向量（Tensor[B]）：每个样本一个 CR
    """
    if isinstance(cr, torch.Tensor):
        if cr.dim() == 0:
            cr = cr.expand(batch_size)
        if cr.shape[0] != batch_size:
            raise ValueError(f"cr tensor batch mismatch: got {cr.shape[0]}, expected {batch_size}")
        return torch.clamp(cr.to(device=device, dtype=torch.float32), 1e-3, 1.0)
    return torch.full((batch_size,), float(max(1e-3, min(1.0, cr))), device=device, dtype=torch.float32)


def _make_topk_mask(weights: torch.Tensor, cr: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    根据通道权重构造硬掩码（Hard Top-k）。

    参数：
    - weights: [B, C]，每个样本每个通道一个权重（0~1）
    - cr: 标量或 [B] 张量，表示“保留比例”

    返回：
    - mask: [B, C]，0/1 掩码

    逻辑：对每个样本保留 top-k（largest=True）权重通道。
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
    """AWGN 信道：y = x + n，噪声方差由 SNR 和 power 决定。"""
    gamma = 10 ** (snr_db / 10.0)
    std = torch.sqrt(torch.tensor(power / gamma, device=x.device, dtype=x.dtype))
    noise = std * torch.randn_like(x)
    return x + noise


def fading_channel(x: torch.Tensor, snr_db: float, power: float = 2.0) -> torch.Tensor:
    """
    平坦瑞利衰落 + 复高斯噪声 + 理想均衡。

    输入：
    - x: [B, F] 实数向量，会按偶/奇位拼成复符号

    输出：
    - y_out: [B, F] 实数向量（复数结果再拆回实部/虚部）
    """
    gamma = 10 ** (snr_db / 10.0)
    b, feature_len = x.shape
    need_pad = feature_len % 2 == 1

    if need_pad:
        # Fading 处理时需要按 2 个实数 -> 1 个复数配对，奇数长度先补 0。
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
    # 理想信道估计下的零迫近均衡（直接除以 h）
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
    """
    组合信道：
    1) 先在展平向量上执行 fading_channel
    2) reshape 回 [B,C,H,W]
    3) 再叠加空间可变 SNR 的 AWGN
    """
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
    将信道状态编码为条件向量。

    输出 cond_vec 维度为 2*embed_dim（默认 32）：
    - channel embedding 分支（离散信道类型）
    - SNR MLP 分支（连续信道强度）
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        # 离散信道类型嵌入：
        # 3 类信道 -> embed_dim 维向量
        # 输出形状: [B, embed_dim]
        self.channel_embed = nn.Embedding(num_embeddings=3, embedding_dim=embed_dim)
        # 连续信道质量（snr）编码：
        # 输入: [B,1]，输出: [B,embed_dim]
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, channel_type: str, snr_db: float, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        返回形状 [B, cond_dim] 的条件向量（默认 cond_dim=32）。

        详细步骤：
        1) 把 channel_type 编成索引并查 embedding -> channel_vec:[B,16]
        2) 把 snr_db 归一化到约 [0,1]，送入 MLP -> snr_vec:[B,16]
        3) 拼接两路向量 -> cond_vec:[B,32]
        """
        channel_idx = _channel_index_tensor(channel_type, batch_size=batch_size, device=device)
        channel_vec = self.channel_embed(channel_idx)

        # 这里用 28 做归一化分母，因为训练中 SNR 基本采样在 [0,28]。
        snr_norm = float(snr_db) / 28.0
        snr_input = torch.full((batch_size, 1), snr_norm, device=device, dtype=torch.float32)
        snr_vec = self.snr_mlp(snr_input)
        return torch.cat([channel_vec, snr_vec], dim=1)


class ChannelConditionedSelector(nn.Module):
    """
    信道条件化的通道选择器。

    输入：
    - semantic_vec: [B, C] 语义统计向量（来自 GAP）
    - cond_vec: [B, cond_dim] 信道条件向量
    输出：
    - weights: [B, C] 通道权重（0~1）
    """

    def __init__(self, channels: int, ratio: int, cond_dim: int = 32):
        super().__init__()
        hidden = max(1, channels // ratio)
        # 语义特征分支: [B,C] -> [B,hidden]
        self.semantic_proj = nn.Linear(channels, hidden, bias=False)
        # 信道条件分支: [B,cond_dim] -> [B,hidden]
        self.cond_proj = nn.Linear(cond_dim, hidden, bias=False)
        # 融合后回投影到通道维: [B,hidden] -> [B,C]
        self.out_proj = nn.Linear(hidden, channels, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, semantic_vec: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        """
        语义分支与信道分支在 hidden 维度相加融合，再映射回通道权重。

        数学形式：
        fused = W_s * semantic_vec + W_c * cond_vec
        weights = sigmoid(W_o * relu(fused))
        """
        fused = self.semantic_proj(semantic_vec) + self.cond_proj(cond_vec)
        fused = self.act(fused)
        return self.gate(self.out_proj(fused))


class RateController(nn.Module):
    """
    动态码率控制器：根据语义 + 信道状态预测样本级 CR。
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
        """
        返回每个样本的 CR（[B]）：
        1) 网络输出 raw in (0,1)
        2) 映射到 [min_cr, max_cr]
        3) 与 base_cr 做线性融合（blend_alpha 控制动态程度）
        """
        raw = self.net(torch.cat([semantic_vec, cond_vec], dim=1)).squeeze(1)
        dynamic_cr = min_cr + (max_cr - min_cr) * raw
        base = torch.full_like(dynamic_cr, float(base_cr))
        blended = (1.0 - blend_alpha) * base + blend_alpha * dynamic_cr
        return torch.clamp(blended, min=min_cr, max=max_cr)


class SearchableSEBlock(nn.Module):
    """
    可搜索 SE 模块（支持信道条件化选择 + 动态 CR）。

    输出：
    - mask 后特征图 [B,C,H,W]
    - 样本级 CR [B]
    """

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

        # 当禁用信道条件时，退化为普通 SE 的两层 FC 选择器。
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
        """
        前向流程：
        1) GAP 得到语义向量 semantic_vec
        2) 信道编码得到 cond_vec
        3) 生成通道权重 weights
        4) 预测样本级 CR（或使用固定 CR）
        5) top-k 生成硬掩码并作用到特征图
        """
        b, c, _, _ = x.shape
        # semantic_vec:[B,C]，每个通道一个全局语义统计值
        semantic_vec = self.gap(x).view(b, c)
        # cond_vec:[B,32]，包含离散信道类型 + 连续SNR信息
        cond_vec = self.condition_encoder(channel_type=channel_type, snr_db=snr_db, batch_size=b, device=x.device)

        if self.use_channel_condition:
            # 条件化选择器：通道权重随“当前信道状态”变化
            weights = self.cond_selector(semantic_vec=semantic_vec, cond_vec=cond_vec)
        else:
            # 退化到普通 SE 权重（不看信道）
            weights = self.base_fc(semantic_vec)

        if self.use_dynamic_rate:
            # 每个样本预测一个 CR_i（样本级动态压缩率）
            cr_values = self.rate_controller(
                semantic_vec=semantic_vec,
                cond_vec=cond_vec,
                base_cr=base_cr,
                min_cr=self.min_dynamic_cr,
                max_cr=self.max_dynamic_cr,
                blend_alpha=self.rate_blend_alpha,
            )
        else:
            # 全部样本使用固定基准压缩率
            cr_values = torch.full((b,), float(base_cr), device=x.device, dtype=x.dtype)

        # 根据每个样本的 CR_i 做 top-k 通道保留，得到硬掩码
        mask = _make_topk_mask(weights, cr=cr_values).view(b, c, 1, 1)
        # 最终输出仍是 [B,C,H,W]，只是部分通道被置零
        return x * mask, cr_values


def _build_channel_schedule(in_ch: int, bottleneck_ch: int, depth: int) -> List[int]:
    """
    构造通道调度表（单调插值）。

    例如 in=512, bottleneck=32, depth=3
    -> [512, 352, 192, 32]（具体数值由线性插值+round决定）
    """
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
    """
    可搜索自编码传输模块。

    结构：
    - Encoder: in_channels -> bottleneck_channels
    - 信道扰动: AWGN/Fading/Combined
    - Decoder: bottleneck_channels -> in_channels
    - 可选残差旁路：out += residual
    """

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
        """
        输入/输出形状保持一致：[B, C, H, W]。
        """
        residual = x
        z = self.encoder(x)
        if self.training and self.latent_noise_std > 0:
            # 训练时在 latent 空间额外加噪，提高鲁棒性。
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
            # 仅当形状一致时启用残差融合，避免尺寸不匹配。
            out = out + residual
        return out


class ChannelAwareClassifier(nn.Module):
    """
    ResNet18 + 可搜索 SE + 可搜索 JSCC 的分类器。

    插入策略：
    - insertion_stage=3：layer3 后插入 JSCC，再过 layer4
    - insertion_stage=4：layer4 后插入 JSCC

    运行时统计：
    - last_forward_stats 记录最近一次前向的 CR 统计（均值/方差/极值）
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

        # layer3 输出通道 256；layer4 输出通道 512。
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
        """缓存当前 batch 的 CR 统计，供训练/评估日志读取。"""
        cr_detached = cr_values.detach().float()
        self.last_forward_stats = {
            "mean_cr": float(cr_detached.mean().item()),
            "std_cr": float(cr_detached.std(unbiased=False).item()),
            "min_cr": float(cr_detached.min().item()),
            "max_cr": float(cr_detached.max().item()),
        }

    def forward(self, x: torch.Tensor, channel_type: str, snr_db: float) -> torch.Tensor:
        """
        标准前向：
        图像 -> ResNet 前半 -> (SE+JSCC) -> ResNet 后半 -> 分类头
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.arch.insertion_stage == 3:
            # 在 layer3 后插入可搜索 SE + JSCC。
            x, cr_values = self.se_block(x, base_cr=self.arch.cr, channel_type=channel_type, snr_db=snr_db)
            self._record_cr_stats(cr_values)
            x = self.jscc_block(x, channel_type=channel_type, snr_db=snr_db)
            x = self.layer4(x)
        else:
            # 在 layer4 后插入可搜索 SE + JSCC。
            x = self.layer4(x)
            x, cr_values = self.se_block(x, base_cr=self.arch.cr, channel_type=channel_type, snr_db=snr_db)
            self._record_cr_stats(cr_values)
            x = self.jscc_block(x, channel_type=channel_type, snr_db=snr_db)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
