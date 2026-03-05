from __future__ import annotations

"""
NAS 候选架构重训练脚本。

用途：
1. 读取 NAS 输出的 best_arch.json（或手工指定架构 JSON）。
2. 在完整训练轮数下进行正式训练。
3. 在多信道/SNR 网格上持续评估并保存最佳模型。
4. 导出训练历史 summary.json。
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nas.nas_utils import (  # noqa: E402
    build_dataloaders,
    default_channel_sampler,
    load_architecture,
    resolve_path,
    save_architecture,
    set_seed,
)
from scripts.nas.searchable_model import ChannelAwareClassifier  # noqa: E402


def parse_int_list(values: str) -> List[int]:
    """将逗号分隔字符串解析为整数列表。"""
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_str_list(values: str) -> List[str]:
    """将逗号分隔字符串解析为字符串列表。"""
    return [v.strip() for v in values.split(",") if v.strip()]


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """
    单轮训练。

    每个 batch 随机采样一个 (channel_type, snr) 条件，
    以提升模型对多信道场景的泛化能力。
    """
    sampler = default_channel_sampler()
    rng = random.Random(seed)
    model.train()
    running_loss = 0.0
    cr_values: List[float] = []

    for images, labels in train_loader:
        channel_type, snr = sampler.sample(rng)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images, channel_type=channel_type, snr_db=float(snr))
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if hasattr(model, "last_forward_stats"):
            cr_values.append(float(model.last_forward_stats.get("mean_cr", 0.0)))

    mean_cr = sum(cr_values) / max(len(cr_values), 1) if cr_values else 0.0
    std_cr = (
        float(torch.tensor(cr_values, dtype=torch.float32).std(unbiased=False).item())
        if len(cr_values) > 1
        else 0.0
    )
    return {
        "train_loss": running_loss / max(len(train_loader), 1),
        "train_mean_cr": mean_cr,
        "train_std_cr": std_cr,
    }


def evaluate_grid(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_channel_types: List[str],
    eval_snr_list: List[int],
) -> Dict[str, object]:
    """
    在验证集上执行“信道类型 × SNR”网格评估。

    返回：
    - mean_acc / worst_acc
    - mean_valid_loss
    - mean_cr / std_cr（动态码率统计）
    - acc_map（每个网格点的准确率）
    """
    model.eval()
    acc_map: Dict[str, float] = {}
    acc_values: List[float] = []
    criterion = nn.CrossEntropyLoss()
    valid_loss_values: List[float] = []
    cr_values: List[float] = []

    with torch.no_grad():
        for channel_type in eval_channel_types:
            for snr in eval_snr_list:
                correct = 0
                total = 0
                loss_sum = 0.0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images, channel_type=channel_type, snr_db=float(snr))
                    loss = criterion(logits, labels)
                    loss_sum += loss.item()
                    pred = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    if hasattr(model, "last_forward_stats"):
                        cr_values.append(float(model.last_forward_stats.get("mean_cr", 0.0)))
                acc = correct / max(total, 1)
                avg_loss = loss_sum / max(len(loader), 1)
                acc_map[f"{channel_type}@{snr}dB"] = acc
                acc_values.append(acc)
                valid_loss_values.append(avg_loss)

    mean_cr = sum(cr_values) / max(len(cr_values), 1) if cr_values else 0.0
    std_cr = (
        float(torch.tensor(cr_values, dtype=torch.float32).std(unbiased=False).item())
        if len(cr_values) > 1
        else 0.0
    )
    return {
        "mean_acc": sum(acc_values) / max(len(acc_values), 1),
        "worst_acc": min(acc_values) if acc_values else 0.0,
        "mean_valid_loss": sum(valid_loss_values) / max(len(valid_loss_values), 1),
        "mean_cr": mean_cr,
        "std_cr": std_cr,
        "acc_map": acc_map,
    }


def parse_args() -> argparse.Namespace:
    """命令行参数定义。"""
    parser = argparse.ArgumentParser(description="Retrain one architecture from NAS outputs.")
    parser.add_argument("--arch_json", type=str, required=True, help="Path to best_arch.json or custom arch json.")
    parser.add_argument("--dataset_name", type=str, default="UCMerced_LandUse")
    parser.add_argument("--train_dir", type=str, default="data/UCMerced_LandUse/UCMerced_LandUse-train")
    parser.add_argument("--valid_dir", type=str, default="data/UCMerced_LandUse/UCMerced_LandUse-valid")
    parser.add_argument("--output_dir", type=str, default="runs/nas_retrain")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--eval_channel_types", type=str, default="AWGN,Fading,Combined_channel")
    parser.add_argument("--eval_snr_list", type=str, default="0,4,8,12,16,20,24,28")
    parser.add_argument("--disable_dynamic_rate", action="store_true")
    parser.add_argument("--disable_channel_condition", action="store_true")
    parser.add_argument("--min_dynamic_cr", type=float, default=0.3)
    parser.add_argument("--max_dynamic_cr", type=float, default=1.0)
    parser.add_argument("--rate_blend_alpha", type=float, default=0.7)
    parser.add_argument("--disable_pretrained_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    重训练入口：
    1) 加载架构
    2) 构建数据与模型
    3) epoch 循环训练 + 网格评估
    4) 保存 best/final 模型与 summary
    """
    args = parse_args()
    set_seed(args.seed)

    arch_path = resolve_path(args.arch_json)
    train_dir = resolve_path(args.train_dir)
    valid_dir = resolve_path(args.valid_dir)
    run_dir = resolve_path(args.output_dir) / f"{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not arch_path.exists():
        raise FileNotFoundError(f"Architecture file not found: {arch_path}")
    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir does not exist: {train_dir}")
    if not valid_dir.exists():
        raise FileNotFoundError(f"Valid dir does not exist: {valid_dir}")

    arch = load_architecture(arch_path)
    save_architecture(run_dir / "arch.json", arch)

    train_loader, valid_loader, class_names = build_dataloaders(
        train_dir=train_dir,
        valid_dir=valid_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    num_classes = len(class_names)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = ChannelAwareClassifier(
        num_classes=num_classes,
        arch=arch,
        pretrained_backbone=not args.disable_pretrained_backbone,
        use_channel_condition=not args.disable_channel_condition,
        use_dynamic_rate=not args.disable_dynamic_rate,
        min_dynamic_cr=args.min_dynamic_cr,
        max_dynamic_cr=args.max_dynamic_cr,
        rate_blend_alpha=args.rate_blend_alpha,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True)

    eval_channel_types = parse_str_list(args.eval_channel_types)
    eval_snr_list = parse_int_list(args.eval_snr_list)

    history: List[Dict[str, object]] = []
    best_metric = -1.0
    best_ckpt_path = run_dir / "best_model.pth"

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            seed=args.seed + epoch,
        )
        valid_result = evaluate_grid(
            model=model,
            loader=valid_loader,
            device=device,
            eval_channel_types=eval_channel_types,
            eval_snr_list=eval_snr_list,
        )

        scheduler.step(float(valid_result["mean_valid_loss"]))
        # 以网格平均验证损失驱动 ReduceLROnPlateau。
        train_loss = float(train_metrics["train_loss"])
        train_mean_cr = float(train_metrics["train_mean_cr"])
        train_std_cr = float(train_metrics["train_std_cr"])
        mean_acc = float(valid_result["mean_acc"])
        worst_acc = float(valid_result["worst_acc"])
        valid_mean_cr = float(valid_result["mean_cr"])
        valid_std_cr = float(valid_result["std_cr"])
        current_lr = optimizer.param_groups[0]["lr"]

        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "mean_valid_loss": valid_result["mean_valid_loss"],
            "mean_acc": mean_acc,
            "worst_acc": worst_acc,
            "train_mean_cr": train_mean_cr,
            "train_std_cr": train_std_cr,
            "valid_mean_cr": valid_mean_cr,
            "valid_std_cr": valid_std_cr,
            "lr": current_lr,
            "acc_map": valid_result["acc_map"],
        }
        history.append(history_row)

        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, mean_acc={mean_acc:.4f}, "
            f"worst_acc={worst_acc:.4f}, valid_mean_cr={valid_mean_cr:.3f}, lr={current_lr:.6f}"
        )

        if mean_acc > best_metric:
            best_metric = mean_acc
            # 以 mean_acc 作为主指标保存最佳模型。
            torch.save(model.state_dict(), best_ckpt_path)

    final_ckpt_path = run_dir / "final_model.pth"
    torch.save(model.state_dict(), final_ckpt_path)

    summary = {
        "dataset_name": args.dataset_name,
        "arch": arch.to_dict(),
        "dynamic_selector": {
            "use_dynamic_rate": not args.disable_dynamic_rate,
            "use_channel_condition": not args.disable_channel_condition,
            "min_dynamic_cr": args.min_dynamic_cr,
            "max_dynamic_cr": args.max_dynamic_cr,
            "rate_blend_alpha": args.rate_blend_alpha,
        },
        "best_mean_acc": best_metric,
        "best_ckpt": str(best_ckpt_path),
        "final_ckpt": str(final_ckpt_path),
        "epochs": args.epochs,
        "history": history,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Retrain complete. Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
