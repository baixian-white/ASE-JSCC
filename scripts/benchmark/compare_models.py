from __future__ import annotations

"""
Compare two checkpoints under a unified evaluation protocol.

The script is intentionally placed in `scripts/benchmark/` so all benchmark
utilities stay isolated from training/search/eval scripts.
"""

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nas.nas_utils import load_architecture, resolve_path  # noqa: E402
from scripts.nas.searchable_model import ChannelAwareClassifier  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_LEGACY_TRAIN_MODULE: ModuleType | None = None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_legacy_train_module() -> ModuleType:
    global _LEGACY_TRAIN_MODULE
    if _LEGACY_TRAIN_MODULE is not None:
        return _LEGACY_TRAIN_MODULE

    train_script = PROJECT_ROOT / "scripts" / "train" / "ASE-JSCCtrain.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Legacy training script not found: {train_script}")

    spec = importlib.util.spec_from_file_location("ase_jscc_legacy_train", str(train_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {train_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LEGACY_TRAIN_MODULE = module
    return module


def awgn_channel(x: torch.Tensor, snr: torch.Tensor | float, device: torch.device, power: float = 2.0) -> torch.Tensor:
    gamma = 10 ** (snr / 10.0)
    noise = torch.sqrt(torch.tensor(power, device=device, dtype=x.dtype) / gamma) * torch.randn_like(x)
    return x + noise


def fading_channel(x: torch.Tensor, snr: torch.Tensor | float, device: torch.device, power: float = 2.0) -> torch.Tensor:
    gamma = 10 ** (snr / 10.0)
    batch_size, feature_length = x.shape
    half = feature_length // 2
    h_i = torch.randn(batch_size, half, device=device, dtype=x.dtype)
    h_r = torch.randn(batch_size, half, device=device, dtype=x.dtype)
    h = torch.complex(h_i, h_r)
    x_c = torch.complex(x[:, 0:feature_length:2], x[:, 1:feature_length:2])
    y_c = h * x_c
    n_i = torch.sqrt(torch.tensor(power, device=device, dtype=x.dtype) / gamma) * torch.randn(batch_size, half, device=device, dtype=x.dtype)
    n_r = torch.sqrt(torch.tensor(power, device=device, dtype=x.dtype) / gamma) * torch.randn(batch_size, half, device=device, dtype=x.dtype)
    noise = torch.complex(n_i, n_r)
    y = (y_c + noise) / h

    y_out = torch.zeros(batch_size, feature_length, device=device, dtype=x.dtype)
    y_out[:, 0:feature_length:2] = y.real
    y_out[:, 1:feature_length:2] = y.imag
    return y_out


def combined_channel(
    x: torch.Tensor,
    snr: torch.Tensor | float,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x_faded = fading_channel(x, snr, device=device)
    x_faded = x_faded.view(batch_size, channels, height, width)
    snr_map = torch.randint(0, 28, (batch_size, channels, height, 1), device=device)
    return awgn_channel(x_faded, snr_map, device=device)


def legacy_channel(
    z: torch.Tensor,
    snr: torch.Tensor | float,
    channel_type: str,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if channel_type == "AWGN":
        return awgn_channel(z, snr, device=device)
    if channel_type == "Fading":
        return fading_channel(z, snr, device=device)
    if channel_type == "Combined_channel":
        return combined_channel(z, snr, batch_size, channels, height, width, device=device)
    raise ValueError(f"Unknown channel type: {channel_type}")


def mask_gen(weights: torch.Tensor, cr: float) -> torch.Tensor:
    position = max(1, round(cr * weights.size(1)))
    weights_sorted, _ = torch.sort(weights, dim=1)
    mask = torch.zeros_like(weights)
    for i in range(weights.size(0)):
        threshold = weights_sorted[i, position - 1]
        for j in range(weights.size(1)):
            if weights[i, j] <= threshold:
                mask[i, j] = 1
    return mask


class LegacySEBlock(nn.Module):
    def __init__(self, in_channels: int, ratio: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cr: float = 0.8) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y)
        mask = mask_gen(y, cr).view(b, c, 1, 1)
        return x * mask


class LegacyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, channel_type: str, device: torch.device) -> torch.Tensor:
        x = self.encoder(x)
        x = x + torch.randn_like(x) * 0.1
        batch_size, channels, height, width = x.shape

        if channel_type in ["Fading", "Combined_channel"]:
            x = self.flatten(x)
            snr = torch.randint(0, 28, (x.shape[0], 1), device=device)
        else:
            snr = torch.randint(0, 28, (x.shape[0], x.shape[1], x.shape[2], 1), device=device)

        x = legacy_channel(
            x,
            snr,
            channel_type=channel_type,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            device=device,
        )
        x = x.view(batch_size, channels, height, width)
        return self.decoder(x)


class LegacySatelliteClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet18 = resnet18(weights=None)
        in_features = self.resnet18.fc.in_features
        self.attention_module = LegacySEBlock(in_features)
        # Keep original typo to match checkpoint key names.
        self.antoencoder = LegacyAutoencoder()
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, cr: float, channel_type: str, device: torch.device) -> torch.Tensor:
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.attention_module(x, cr)
        x = self.antoencoder(x, channel_type, device)
        x = self.resnet18.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.resnet18.fc(x)


@dataclass
class ModelSpec:
    name: str
    model_type: str
    ckpt_path: Path
    arch_json: Path | None = None
    legacy_cr: float = 0.8
    disable_dynamic_rate: bool = False
    disable_channel_condition: bool = False
    min_dynamic_cr: float = 0.3
    max_dynamic_cr: float = 1.0
    rate_blend_alpha: float = 0.7
    disable_pretrained_backbone: bool = False


def _add_model_args(parser: argparse.ArgumentParser, prefix: str, default_name: str) -> None:
    parser.add_argument(f"--{prefix}_name", type=str, default=default_name)
    parser.add_argument(f"--{prefix}_type", type=str, choices=["legacy", "nas"], required=True)
    parser.add_argument(f"--{prefix}_ckpt", type=str, required=True)
    parser.add_argument(f"--{prefix}_arch_json", type=str, default="")
    parser.add_argument(f"--{prefix}_legacy_cr", type=float, default=0.8)
    parser.add_argument(f"--{prefix}_disable_dynamic_rate", action="store_true")
    parser.add_argument(f"--{prefix}_disable_channel_condition", action="store_true")
    parser.add_argument(f"--{prefix}_min_dynamic_cr", type=float, default=0.3)
    parser.add_argument(f"--{prefix}_max_dynamic_cr", type=float, default=1.0)
    parser.add_argument(f"--{prefix}_rate_blend_alpha", type=float, default=0.7)
    parser.add_argument(f"--{prefix}_disable_pretrained_backbone", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark two models under unified channel-aware evaluation.")
    parser.add_argument("--data_dir", type=str, required=True, help="ImageFolder directory for benchmarking.")
    parser.add_argument("--output_dir", type=str, default="runs/model_benchmark")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc_runs", type=int, default=5, help="Monte-Carlo runs per channel type.")
    parser.add_argument("--channel_types", type=str, default="AWGN,Fading,Combined_channel")
    parser.add_argument("--nas_eval_snr_min", type=int, default=0)
    parser.add_argument("--nas_eval_snr_max", type=int, default=28)

    _add_model_args(parser, "model_a", "baseline")
    _add_model_args(parser, "model_b", "candidate")
    return parser.parse_args()


def _parse_model_spec(args: argparse.Namespace, prefix: str) -> ModelSpec:
    arch_json = getattr(args, f"{prefix}_arch_json")
    return ModelSpec(
        name=getattr(args, f"{prefix}_name"),
        model_type=getattr(args, f"{prefix}_type"),
        ckpt_path=resolve_path(getattr(args, f"{prefix}_ckpt")),
        arch_json=resolve_path(arch_json) if arch_json else None,
        legacy_cr=float(getattr(args, f"{prefix}_legacy_cr")),
        disable_dynamic_rate=bool(getattr(args, f"{prefix}_disable_dynamic_rate")),
        disable_channel_condition=bool(getattr(args, f"{prefix}_disable_channel_condition")),
        min_dynamic_cr=float(getattr(args, f"{prefix}_min_dynamic_cr")),
        max_dynamic_cr=float(getattr(args, f"{prefix}_max_dynamic_cr")),
        rate_blend_alpha=float(getattr(args, f"{prefix}_rate_blend_alpha")),
        disable_pretrained_backbone=bool(getattr(args, f"{prefix}_disable_pretrained_backbone")),
    )


def build_eval_loader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, List[str], int]:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, dataset.classes, len(dataset)


def build_model_and_forward(
    spec: ModelSpec,
    num_classes: int,
    device: torch.device,
    nas_eval_snr_min: int,
    nas_eval_snr_max: int,
) -> Tuple[nn.Module, Callable[[torch.Tensor, str, random.Random], torch.Tensor]]:
    if spec.model_type == "nas":
        if spec.arch_json is None:
            raise ValueError(f"{spec.name}: NAS model requires --arch_json.")
        arch = load_architecture(spec.arch_json)
        model = ChannelAwareClassifier(
            num_classes=num_classes,
            arch=arch,
            pretrained_backbone=not spec.disable_pretrained_backbone,
            use_channel_condition=not spec.disable_channel_condition,
            use_dynamic_rate=not spec.disable_dynamic_rate,
            min_dynamic_cr=spec.min_dynamic_cr,
            max_dynamic_cr=spec.max_dynamic_cr,
            rate_blend_alpha=spec.rate_blend_alpha,
        ).to(device)
        state = torch.load(str(spec.ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()

        def forward_fn(images: torch.Tensor, channel_type: str, rng: random.Random) -> torch.Tensor:
            snr = rng.randint(nas_eval_snr_min, nas_eval_snr_max)
            return model(images, channel_type=channel_type, snr_db=float(snr))

        return model, forward_fn

    if spec.model_type == "legacy":
        legacy_module = _load_legacy_train_module()
        if not hasattr(legacy_module, "SatelliteClassifierWithAttention"):
            raise AttributeError(
                "Legacy training module missing `SatelliteClassifierWithAttention`."
            )
        # The original legacy channel functions read this module-level variable.
        setattr(legacy_module, "device", device)
        model = legacy_module.SatelliteClassifierWithAttention(num_classes).to(device)
        state = torch.load(str(spec.ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.eval()

        def forward_fn(images: torch.Tensor, channel_type: str, rng: random.Random) -> torch.Tensor:
            _ = rng  # legacy model samples SNR internally via torch.randint
            return model(images, cr=float(spec.legacy_cr), channel_type=channel_type)

        return model, forward_fn

    raise ValueError(f"Unknown model type: {spec.model_type}")


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95": 0.0}
    mean_val = float(sum(values) / len(values))
    if len(values) > 1:
        var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
        std_val = float(math.sqrt(var))
    else:
        std_val = 0.0
    ci95 = float(1.96 * std_val / math.sqrt(max(1, len(values))))
    return {
        "mean": mean_val,
        "std": std_val,
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95": ci95,
    }


def evaluate_model(
    model_name: str,
    model: nn.Module,
    forward_fn: Callable[[torch.Tensor, str, random.Random], torch.Tensor],
    loader: DataLoader,
    device: torch.device,
    channel_types: List[str],
    mc_runs: int,
    seed: int,
) -> Dict[str, object]:
    del model
    channel_runs: Dict[str, List[float]] = {channel: [] for channel in channel_types}
    with torch.no_grad():
        for c_idx, channel in enumerate(channel_types):
            for run_idx in range(mc_runs):
                local_seed = seed + c_idx * 10_000 + run_idx
                _seed_all(local_seed)
                rng = random.Random(local_seed)
                correct = 0
                total = 0
                for images, labels in loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = forward_fn(images, channel, rng)
                    pred = torch.argmax(logits, dim=1)
                    correct += int((pred == labels).sum().item())
                    total += int(labels.size(0))
                channel_runs[channel].append(float(correct / max(total, 1)))

    per_channel = {channel: {"runs": runs, **_stats(runs)} for channel, runs in channel_runs.items()}
    overall_runs = [
        float(sum(channel_runs[channel][run_idx] for channel in channel_types) / len(channel_types))
        for run_idx in range(mc_runs)
    ]
    overall_mean = float(sum(per_channel[ch]["mean"] for ch in channel_types) / len(channel_types))
    overall_worst = float(min(per_channel[ch]["mean"] for ch in channel_types))
    return {
        "model_name": model_name,
        "per_channel": per_channel,
        "overall": {
            "mean_acc": overall_mean,
            "worst_channel_acc": overall_worst,
            "run_mean_acc": overall_runs,
            **_stats(overall_runs),
        },
    }


def pairwise_delta(result_a: Dict[str, object], result_b: Dict[str, object], channel_types: List[str]) -> Dict[str, object]:
    deltas: Dict[str, object] = {}
    for channel in channel_types:
        runs_a = result_a["per_channel"][channel]["runs"]
        runs_b = result_b["per_channel"][channel]["runs"]
        diff_runs = [float(b - a) for a, b in zip(runs_a, runs_b)]
        deltas[channel] = {"runs": diff_runs, **_stats(diff_runs)}

    overall_diff_runs = [
        float(b - a)
        for a, b in zip(result_a["overall"]["run_mean_acc"], result_b["overall"]["run_mean_acc"])
    ]
    deltas["overall_mean"] = {"runs": overall_diff_runs, **_stats(overall_diff_runs)}
    return deltas


def save_csv(path: Path, channel_types: List[str], result_a: Dict[str, object], result_b: Dict[str, object], delta: Dict[str, object]) -> None:
    headers = [
        "scope",
        "model_a_mean",
        "model_a_std",
        "model_b_mean",
        "model_b_std",
        "delta_b_minus_a_mean",
        "delta_b_minus_a_ci95",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for scope in [*channel_types, "overall_mean"]:
            if scope == "overall_mean":
                a_mean = result_a["overall"]["mean"]
                a_std = result_a["overall"]["std"]
                b_mean = result_b["overall"]["mean"]
                b_std = result_b["overall"]["std"]
            else:
                a_mean = result_a["per_channel"][scope]["mean"]
                a_std = result_a["per_channel"][scope]["std"]
                b_mean = result_b["per_channel"][scope]["mean"]
                b_std = result_b["per_channel"][scope]["std"]
            writer.writerow(
                [
                    scope,
                    f"{a_mean:.6f}",
                    f"{a_std:.6f}",
                    f"{b_mean:.6f}",
                    f"{b_std:.6f}",
                    f"{delta[scope]['mean']:.6f}",
                    f"{delta[scope]['ci95']:.6f}",
                ]
            )


def save_markdown(path: Path, spec_a: ModelSpec, spec_b: ModelSpec, channel_types: List[str], result_a: Dict[str, object], result_b: Dict[str, object], delta: Dict[str, object]) -> None:
    lines = [
        "# Model Benchmark Summary",
        "",
        f"- model_a: `{spec_a.name}` ({spec_a.model_type})",
        f"- model_b: `{spec_b.name}` ({spec_b.model_type})",
        "",
        "| scope | A mean±std | B mean±std | delta(B-A) | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in [*channel_types, "overall_mean"]:
        if scope == "overall_mean":
            a_mean = result_a["overall"]["mean"]
            a_std = result_a["overall"]["std"]
            b_mean = result_b["overall"]["mean"]
            b_std = result_b["overall"]["std"]
        else:
            a_mean = result_a["per_channel"][scope]["mean"]
            a_std = result_a["per_channel"][scope]["std"]
            b_mean = result_b["per_channel"][scope]["mean"]
            b_std = result_b["per_channel"][scope]["std"]
        lines.append(
            f"| {scope} | {a_mean:.4f}±{a_std:.4f} | {b_mean:.4f}±{b_std:.4f} | {delta[scope]['mean']:.4f} | ±{delta[scope]['ci95']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_channel_bars(path: Path, model_a_name: str, model_b_name: str, channel_types: List[str], result_a: Dict[str, object], result_b: Dict[str, object]) -> None:
    x = list(range(len(channel_types)))
    width = 0.38
    a_means = [result_a["per_channel"][c]["mean"] for c in channel_types]
    a_stds = [result_a["per_channel"][c]["std"] for c in channel_types]
    b_means = [result_b["per_channel"][c]["mean"] for c in channel_types]
    b_stds = [result_b["per_channel"][c]["std"] for c in channel_types]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.bar([v - width / 2 for v in x], a_means, width=width, yerr=a_stds, capsize=4, label=model_a_name, color="#4c78a8")
    ax.bar([v + width / 2 for v in x], b_means, width=width, yerr=b_stds, capsize=4, label=model_b_name, color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_types)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Channel Accuracy Comparison")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def plot_delta(path: Path, channel_types: List[str], delta: Dict[str, object]) -> None:
    scopes = [*channel_types, "overall_mean"]
    means = [delta[s]["mean"] for s in scopes]
    cis = [delta[s]["ci95"] for s in scopes]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in means]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.bar(scopes, means, yerr=cis, capsize=4, color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Accuracy Delta (B - A)")
    ax.set_title("Delta with 95% CI")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    channel_types = [v.strip() for v in args.channel_types.split(",") if v.strip()]
    if not channel_types:
        raise ValueError("channel_types must contain at least one item.")
    if args.mc_runs < 1:
        raise ValueError("mc_runs must be >= 1.")
    if args.nas_eval_snr_min > args.nas_eval_snr_max:
        raise ValueError("nas_eval_snr_min must be <= nas_eval_snr_max.")

    model_a = _parse_model_spec(args, "model_a")
    model_b = _parse_model_spec(args, "model_b")
    eval_dir = resolve_path(args.data_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {eval_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader, class_names, eval_size = build_eval_loader(
        data_dir=eval_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() or f"{model_a.name}_vs_{model_b.name}_{timestamp}"
    out_dir = resolve_path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_a_obj, forward_a = build_model_and_forward(
        spec=model_a,
        num_classes=len(class_names),
        device=device,
        nas_eval_snr_min=args.nas_eval_snr_min,
        nas_eval_snr_max=args.nas_eval_snr_max,
    )
    model_b_obj, forward_b = build_model_and_forward(
        spec=model_b,
        num_classes=len(class_names),
        device=device,
        nas_eval_snr_min=args.nas_eval_snr_min,
        nas_eval_snr_max=args.nas_eval_snr_max,
    )

    print(f"Benchmark output dir: {out_dir}")
    print(f"Device: {device}")
    print(f"Eval samples: {eval_size}, classes: {len(class_names)}")
    print(f"Model A: {model_a.name} ({model_a.model_type})")
    print(f"Model B: {model_b.name} ({model_b.model_type})")

    result_a = evaluate_model(
        model_name=model_a.name,
        model=model_a_obj,
        forward_fn=forward_a,
        loader=loader,
        device=device,
        channel_types=channel_types,
        mc_runs=args.mc_runs,
        seed=args.seed,
    )
    result_b = evaluate_model(
        model_name=model_b.name,
        model=model_b_obj,
        forward_fn=forward_b,
        loader=loader,
        device=device,
        channel_types=channel_types,
        mc_runs=args.mc_runs,
        seed=args.seed,
    )
    delta = pairwise_delta(result_a, result_b, channel_types)

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "eval_data_dir": str(eval_dir),
            "output_dir": str(out_dir),
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "seed": args.seed,
            "mc_runs": args.mc_runs,
            "channel_types": channel_types,
            "nas_eval_snr_min": args.nas_eval_snr_min,
            "nas_eval_snr_max": args.nas_eval_snr_max,
        },
        "models": {
            "model_a": {
                "name": model_a.name,
                "type": model_a.model_type,
                "ckpt": str(model_a.ckpt_path),
                "arch_json": str(model_a.arch_json) if model_a.arch_json else None,
            },
            "model_b": {
                "name": model_b.name,
                "type": model_b.model_type,
                "ckpt": str(model_b.ckpt_path),
                "arch_json": str(model_b.arch_json) if model_b.arch_json else None,
            },
        },
        "results": {
            "model_a": result_a,
            "model_b": result_b,
            "delta_b_minus_a": delta,
        },
    }

    (out_dir / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_csv(out_dir / "benchmark_table.csv", channel_types, result_a, result_b, delta)
    save_markdown(out_dir / "benchmark_summary.md", model_a, model_b, channel_types, result_a, result_b, delta)
    plot_channel_bars(out_dir / "fig_channel_acc_compare.png", model_a.name, model_b.name, channel_types, result_a, result_b)
    plot_delta(out_dir / "fig_delta_ci95.png", channel_types, delta)

    print(f"Done. Summary: {out_dir / 'benchmark_summary.md'}")
    print(f"JSON: {out_dir / 'benchmark_results.json'}")
    print(f"CSV: {out_dir / 'benchmark_table.csv'}")


if __name__ == "__main__":
    main()
