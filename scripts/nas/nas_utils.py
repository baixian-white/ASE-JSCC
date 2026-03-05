from __future__ import annotations

"""
NAS 公共工具函数。

职责分层：
1. 路径与随机性工具：项目根目录定位、路径解析、随机种子设置。
2. 数据工具：构建训练/验证 transforms 与 dataloader。
3. 搜索评估工具：参数量、传输代价估计、鲁棒性统计。
4. JSON 持久化：架构与搜索结果读写。
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scripts.nas.search_space import ArchitectureConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_project_root() -> Path:
    """
    向上查找包含 .git 的目录作为项目根目录。

    这样脚本可以从任意 cwd 启动，而不是强依赖“在仓库根目录执行”。
    """
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def resolve_path(path_value: str) -> Path:
    """
    解析输入路径：
    - 绝对路径：原样返回
    - 相对路径：拼接到 PROJECT_ROOT 下
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """
    固定所有常见随机源，尽可能提高可复现性。

    说明：开启 cudnn.deterministic 后，速度可能略慢，但实验更稳定。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 256) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    构建训练/验证预处理。

    训练集使用常见增强（翻转、颜色扰动），验证集仅做 resize+normalize。
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


def build_dataloaders(
    train_dir: Path,
    valid_dir: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    image_size: int = 256,
) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    """
    基于 ImageFolder 构建 train/valid dataloader，并返回类别名顺序。

    返回值：
    - train_loader
    - valid_loader
    - classes（用于确定分类头 num_classes）
    """
    train_transform, valid_transform = build_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=str(valid_dir), transform=valid_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, train_dataset.classes


@dataclass
class ChannelSampler:
    """
    信道采样器：
    - 按指定概率采样信道类型
    - 在 [snr_min, snr_max] 区间采样整数 SNR（dB）
    """

    channel_types: List[str]
    channel_probs: List[float]
    snr_min: int
    snr_max: int

    def sample(self, rng: random.Random) -> Tuple[str, int]:
        """采样一次 (channel_type, snr_db)。"""
        channel_type = rng.choices(self.channel_types, weights=self.channel_probs, k=1)[0]
        snr = rng.randint(self.snr_min, self.snr_max)
        return channel_type, snr


def default_channel_sampler() -> ChannelSampler:
    """默认信道分布：三类信道近似均匀。"""
    return ChannelSampler(
        channel_types=["AWGN", "Fading", "Combined_channel"],
        channel_probs=[0.33, 0.33, 0.34],
        snr_min=0,
        snr_max=28,
    )


def parameter_count_m(model: torch.nn.Module) -> float:
    """统计可训练参数量，单位 Million（M）。"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(total) / 1_000_000.0


def estimate_tx_cost(
    arch: ArchitectureConfig,
    image_size: int = 256,
    effective_cr: float | None = None,
) -> float:
    """
    粗略估计传输代价（符号数）。

    估计逻辑：
    - 以插入层输出特征图空间尺寸为基础（layer3: /16, layer4: /32）
    - 传输量 ~ bottleneck_channels * H * W * CR
    - 用于多目标比较（相对量纲），不是严格物理仿真值
    """
    # ResNet18 特征图尺寸估计：
    # layer3 -> image_size / 16, layer4 -> image_size / 32
    if arch.insertion_stage == 3:
        h = image_size // 16
        w = image_size // 16
    else:
        h = image_size // 32
        w = image_size // 32
    cr_value = float(arch.cr if effective_cr is None else effective_cr)
    return float(arch.bottleneck_channels * h * w * cr_value)


def robust_gap(acc_values: Iterable[float]) -> float:
    """
    鲁棒性差距：max(acc) - min(acc)。
    值越小表示跨信道/SNR性能波动越小，鲁棒性越好。
    """
    values = list(acc_values)
    if not values:
        return 0.0
    return max(values) - min(values)


def load_architecture(path: Path) -> ArchitectureConfig:
    """从 JSON 文件读取架构配置。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ArchitectureConfig.from_dict(payload)


def save_architecture(path: Path, arch: ArchitectureConfig) -> None:
    """将架构配置保存为 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arch.to_dict(), indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    """向 JSONL 文件追加一行记录（常用于搜索日志流式写入）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
