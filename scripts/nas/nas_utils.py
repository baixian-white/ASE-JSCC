from __future__ import annotations

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
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = get_project_root()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int = 256) -> Tuple[transforms.Compose, transforms.Compose]:
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
    channel_types: List[str]
    channel_probs: List[float]
    snr_min: int
    snr_max: int

    def sample(self, rng: random.Random) -> Tuple[str, int]:
        channel_type = rng.choices(self.channel_types, weights=self.channel_probs, k=1)[0]
        snr = rng.randint(self.snr_min, self.snr_max)
        return channel_type, snr


def default_channel_sampler() -> ChannelSampler:
    return ChannelSampler(
        channel_types=["AWGN", "Fading", "Combined_channel"],
        channel_probs=[0.33, 0.33, 0.34],
        snr_min=0,
        snr_max=28,
    )


def parameter_count_m(model: torch.nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(total) / 1_000_000.0


def estimate_tx_cost(
    arch: ArchitectureConfig,
    image_size: int = 256,
    effective_cr: float | None = None,
) -> float:
    # Feature map size in ResNet18:
    # layer3 output: image_size / 16, layer4 output: image_size / 32
    if arch.insertion_stage == 3:
        h = image_size // 16
        w = image_size // 16
    else:
        h = image_size // 32
        w = image_size // 32
    cr_value = float(arch.cr if effective_cr is None else effective_cr)
    return float(arch.bottleneck_channels * h * w * cr_value)


def robust_gap(acc_values: Iterable[float]) -> float:
    values = list(acc_values)
    if not values:
        return 0.0
    return max(values) - min(values)


def load_architecture(path: Path) -> ArchitectureConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ArchitectureConfig.from_dict(payload)


def save_architecture(path: Path, arch: ArchitectureConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(arch.to_dict(), indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
