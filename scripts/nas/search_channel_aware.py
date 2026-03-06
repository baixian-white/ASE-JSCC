from __future__ import annotations

"""
信道感知 NAS 搜索主脚本。

总体流程：
1. 从搜索空间随机采样候选架构。
2. 对每个候选做少量快速训练（proxy training）。
3. 在多信道、多 SNR 网格上评估性能与鲁棒性。
4. 结合精度、参数量、传输代价、鲁棒性、码率约束计算综合得分。
5. 输出 best_arch / top-k / 搜索日志 JSONL。
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - 可选依赖，缺失时降级
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nas.nas_utils import (  # noqa: E402
    append_jsonl,
    build_dataloaders,
    default_channel_sampler,
    estimate_tx_cost,
    parameter_count_m,
    resolve_path,
    robust_gap,
    save_architecture,
    set_seed,
)
from scripts.nas.search_space import ArchitectureConfig, SearchSpace  # noqa: E402
from scripts.nas.searchable_model import ChannelAwareClassifier  # noqa: E402


def parse_int_list(values: str) -> List[int]:
    """将逗号分隔字符串解析为整数列表，例如 '0,4,8' -> [0,4,8]。"""
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_str_list(values: str) -> List[str]:
    """将逗号分隔字符串解析为字符串列表。"""
    return [v.strip() for v in values.split(",") if v.strip()]


def evaluate_model(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_channel_types: List[str],
    eval_snr_list: List[int],
    max_eval_batches: int,
) -> Dict[str, object]:
    """
    在验证集上做“信道类型 × SNR”网格评估。

    返回：
    - mean_acc / worst_acc / robust_gap
    - mean_cr / std_cr（若模型支持动态 CR 统计）
    - acc_map（每个条件点的准确率）
    """
    model.eval()
    eval_start = time.time()
    acc_map: Dict[str, float] = {}
    acc_values: List[float] = []
    cr_values: List[float] = []
    condition_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for channel_type in eval_channel_types:
            for snr in eval_snr_list:
                correct = 0
                total = 0
                seen_batches = 0
                for batch_idx, (images, labels) in enumerate(valid_loader):
                    if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                        # 可选截断评估 batch，用于加速 NAS 过程。
                        break
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images, channel_type=channel_type, snr_db=float(snr))
                    pred = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    seen_batches += 1
                    if hasattr(model, "last_forward_stats"):
                        cr_values.append(float(model.last_forward_stats.get("mean_cr", 0.0)))
                acc = correct / max(total, 1)
                key = f"{channel_type}@{snr}dB"
                acc_map[key] = acc
                acc_values.append(acc)
                condition_rows.append(
                    {
                        "channel_type": channel_type,
                        "snr_db": int(snr),
                        "acc": float(acc),
                        "correct": int(correct),
                        "total": int(total),
                        "seen_batches": int(seen_batches),
                    }
                )

    mean_acc = sum(acc_values) / max(len(acc_values), 1)
    mean_cr = sum(cr_values) / max(len(cr_values), 1) if cr_values else 0.0
    std_cr = (
        float(torch.tensor(cr_values, dtype=torch.float32).std(unbiased=False).item())
        if len(cr_values) > 1
        else 0.0
    )
    per_channel_mean_acc = {
        channel_type: float(
            sum(row["acc"] for row in condition_rows if row["channel_type"] == channel_type)
            / max(1, sum(1 for row in condition_rows if row["channel_type"] == channel_type))
        )
        for channel_type in eval_channel_types
    }
    per_snr_mean_acc = {
        f"{snr}dB": float(
            sum(row["acc"] for row in condition_rows if int(row["snr_db"]) == int(snr))
            / max(1, sum(1 for row in condition_rows if int(row["snr_db"]) == int(snr)))
        )
        for snr in eval_snr_list
    }
    return {
        "mean_acc": mean_acc,
        "worst_acc": min(acc_values) if acc_values else 0.0,
        "robust_gap": robust_gap(acc_values),
        "mean_cr": mean_cr,
        "std_cr": std_cr,
        "acc_map": acc_map,
        "condition_rows": condition_rows,
        "per_channel_mean_acc": per_channel_mean_acc,
        "per_snr_mean_acc": per_snr_mean_acc,
        "eval_time_sec": float(time.time() - eval_start),
    }


def quick_train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    search_epochs: int,
    learning_rate: float,
    max_train_batches: int,
    seed: int,
) -> Dict[str, object]:
    """
    候选模型的快速代理训练（非最终收敛训练）。

    设计目标：用很低成本粗筛架构，不追求最终精度上限。
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sampler = default_channel_sampler()
    rng = random.Random(seed)

    train_start = time.time()
    model.train()
    epoch_rows: List[Dict[str, object]] = []
    for epoch in range(search_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        seen_batches = 0
        channel_hist = {"AWGN": 0, "Fading": 0, "Combined_channel": 0}
        sampled_snr: List[int] = []
        cr_values: List[float] = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            channel_type, snr = sampler.sample(rng)
            channel_hist[channel_type] = int(channel_hist.get(channel_type, 0)) + 1
            sampled_snr.append(int(snr))
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images, channel_type=channel_type, snr_db=float(snr))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(logits, dim=1)
            running_correct += (pred == labels).sum().item()
            running_total += labels.size(0)
            running_loss += float(loss.item())
            seen_batches += 1
            if hasattr(model, "last_forward_stats"):
                cr_values.append(float(model.last_forward_stats.get("mean_cr", 0.0)))

        mean_cr = sum(cr_values) / max(1, len(cr_values)) if cr_values else 0.0
        std_cr = (
            float(torch.tensor(cr_values, dtype=torch.float32).std(unbiased=False).item())
            if len(cr_values) > 1
            else 0.0
        )
        epoch_rows.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(running_loss / max(seen_batches, 1)),
                "train_acc": float(running_correct / max(running_total, 1)),
                "train_mean_cr": float(mean_cr),
                "train_std_cr": float(std_cr),
                "seen_batches": int(seen_batches),
                "sampled_snr_min": int(min(sampled_snr)) if sampled_snr else None,
                "sampled_snr_max": int(max(sampled_snr)) if sampled_snr else None,
                "sampled_snr_mean": float(sum(sampled_snr) / max(1, len(sampled_snr))) if sampled_snr else None,
                "sampled_channel_hist": channel_hist,
                "epoch_time_sec": float(time.time() - epoch_start),
            }
        )

    return {
        "epochs": epoch_rows,
        "quick_train_time_sec": float(time.time() - train_start),
        "last_epoch": epoch_rows[-1] if epoch_rows else {},
    }


def score_architecture(
    mean_acc: float,
    param_m: float,
    tx_cost: float,
    robust_gap_val: float,
    mean_cr: float,
    lambda_param: float,
    lambda_tx: float,
    lambda_robust: float,
    lambda_cr: float,
    target_cr: float,
) -> float:
    """
    多目标打分函数（单标量）。

    score 越大越好：
    score = mean_acc
            - λ_param * param_m
            - λ_tx * tx_norm
            - λ_robust * robust_gap
            - λ_cr * max(0, mean_cr - target_cr)
    """
    # 将 tx_cost 缩放到和参数量相近的量级，避免单项主导。
    tx_norm = tx_cost / 10_000.0
    cr_violation = max(0.0, mean_cr - target_cr)
    return (
        mean_acc
        - lambda_param * param_m
        - lambda_tx * tx_norm
        - lambda_robust * robust_gap_val
        - lambda_cr * cr_violation
    )


def search_once(
    arch: ArchitectureConfig,
    num_classes: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    local_seed: int,
) -> Dict[str, object]:
    """
    对单个架构执行“一次完整搜索评估”：
    - 建模
    - quick_train
    - 网格评估
    - 代价与综合分数计算
    """
    total_start = time.time()
    build_start = time.time()
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
    build_time_sec = float(time.time() - build_start)

    if device.type == "cuda":
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            # 某些环境下可能拿不到 peak memory，失败时忽略。
            pass

    quick_train_result = quick_train(
        model=model,
        train_loader=train_loader,
        device=device,
        search_epochs=args.search_epochs,
        learning_rate=args.search_lr,
        max_train_batches=args.max_train_batches,
        seed=local_seed,
    )

    eval_start = time.time()
    eval_result = evaluate_model(
        model=model,
        valid_loader=valid_loader,
        device=device,
        eval_channel_types=parse_str_list(args.eval_channel_types),
        eval_snr_list=parse_int_list(args.eval_snr_list),
        max_eval_batches=args.max_eval_batches,
    )
    eval_time_sec = float(time.time() - eval_start)

    param_m = parameter_count_m(model)
    tx_cost = estimate_tx_cost(
        arch,
        image_size=args.image_size,
        effective_cr=float(eval_result["mean_cr"]),
    )
    score = score_architecture(
        mean_acc=float(eval_result["mean_acc"]),
        param_m=param_m,
        tx_cost=tx_cost,
        robust_gap_val=float(eval_result["robust_gap"]),
        mean_cr=float(eval_result["mean_cr"]),
        lambda_param=args.lambda_param,
        lambda_tx=args.lambda_tx,
        lambda_robust=args.lambda_robust,
        lambda_cr=args.lambda_cr,
        target_cr=args.target_cr,
    )
    tx_norm = tx_cost / 10_000.0
    cr_violation = max(0.0, float(eval_result["mean_cr"]) - args.target_cr)

    peak_gpu_mem_mb = None
    if device.type == "cuda":
        try:
            peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        except RuntimeError:
            peak_gpu_mem_mb = None

    return {
        "arch": arch.to_dict(),
        "arch_tag": arch.tag,
        "score": score,
        "mean_acc": eval_result["mean_acc"],
        "worst_acc": eval_result["worst_acc"],
        "robust_gap": eval_result["robust_gap"],
        "mean_cr": eval_result["mean_cr"],
        "std_cr": eval_result["std_cr"],
        "param_m": param_m,
        "tx_cost": tx_cost,
        "acc_map": eval_result["acc_map"],
        "per_channel_mean_acc": eval_result["per_channel_mean_acc"],
        "per_snr_mean_acc": eval_result["per_snr_mean_acc"],
        "condition_rows": eval_result["condition_rows"],
        "quick_train": quick_train_result,
        "score_breakdown": {
            "mean_acc_term": float(eval_result["mean_acc"]),
            "param_penalty": float(-args.lambda_param * param_m),
            "tx_penalty": float(-args.lambda_tx * tx_norm),
            "robust_penalty": float(-args.lambda_robust * float(eval_result["robust_gap"])),
            "cr_penalty": float(-args.lambda_cr * cr_violation),
            "cr_violation": float(cr_violation),
            # Backward-compatible aliases for older analysis code.
            "rate_penalty": float(-args.lambda_cr * cr_violation),
            "rate_violation": float(cr_violation),
            "tx_norm": float(tx_norm),
        },
        "timing": {
            "build_time_sec": build_time_sec,
            "quick_train_time_sec": float(quick_train_result["quick_train_time_sec"]),
            "eval_time_sec": eval_time_sec,
            "total_time_sec": float(time.time() - total_start),
            "eval_time_from_eval_fn_sec": float(eval_result["eval_time_sec"]),
        },
        "resource": {
            "device": str(device),
            "peak_gpu_mem_mb": peak_gpu_mem_mb,
        },
    }


def _search_space_size(search_space: SearchSpace) -> int:
    """计算离散搜索空间全组合数量。"""
    return (
        len(search_space.insertion_stage)
        * len(search_space.se_ratio)
        * len(search_space.cr)
        * len(search_space.bottleneck_channels)
        * len(search_space.ae_depth)
        * len(search_space.kernel_size)
        * len(search_space.use_skip)
    )


def _write_ranked_csv(path: Path, ranked_rows: List[Dict[str, object]]) -> None:
    """将排序后的候选结果导出为 CSV，便于表格分析与画图。"""
    fieldnames = [
        "rank",
        "arch_tag",
        "score",
        "mean_acc",
        "worst_acc",
        "robust_gap",
        "mean_cr",
        "std_cr",
        "param_m",
        "tx_cost",
        "insertion_stage",
        "se_ratio",
        "cr",
        "bottleneck_channels",
        "ae_depth",
        "kernel_size",
        "use_skip",
        "build_time_sec",
        "quick_train_time_sec",
        "eval_time_sec",
        "total_time_sec",
        "peak_gpu_mem_mb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(ranked_rows, start=1):
            arch = row["arch"]
            timing = row.get("timing", {})
            resource = row.get("resource", {})
            writer.writerow(
                {
                    "rank": idx,
                    "arch_tag": row["arch_tag"],
                    "score": row["score"],
                    "mean_acc": row["mean_acc"],
                    "worst_acc": row["worst_acc"],
                    "robust_gap": row["robust_gap"],
                    "mean_cr": row["mean_cr"],
                    "std_cr": row["std_cr"],
                    "param_m": row["param_m"],
                    "tx_cost": row["tx_cost"],
                    "insertion_stage": arch["insertion_stage"],
                    "se_ratio": arch["se_ratio"],
                    "cr": arch["cr"],
                    "bottleneck_channels": arch["bottleneck_channels"],
                    "ae_depth": arch["ae_depth"],
                    "kernel_size": arch["kernel_size"],
                    "use_skip": arch["use_skip"],
                    "build_time_sec": timing.get("build_time_sec", 0.0),
                    "quick_train_time_sec": timing.get("quick_train_time_sec", 0.0),
                    "eval_time_sec": timing.get("eval_time_sec", 0.0),
                    "total_time_sec": timing.get("total_time_sec", 0.0),
                    "peak_gpu_mem_mb": resource.get("peak_gpu_mem_mb", ""),
                }
            )


def _write_topk_markdown(path: Path, top_k: List[Dict[str, object]]) -> None:
    """导出 Top-K Markdown 表，方便直接贴到汇报文档。"""
    lines = [
        "# NAS Search Top-K Summary",
        "",
        "| rank | arch_tag | score | mean_acc | worst_acc | mean_cr | param_m | tx_cost | total_time_sec |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top_k, start=1):
        lines.append(
            "| {rank} | {tag} | {score:.4f} | {mean:.4f} | {worst:.4f} | {mcr:.4f} | {pm:.3f} | {tx:.1f} | {tt:.2f} |".format(
                rank=rank,
                tag=row["arch_tag"],
                score=float(row["score"]),
                mean=float(row["mean_acc"]),
                worst=float(row["worst_acc"]),
                mcr=float(row["mean_cr"]),
                pm=float(row["param_m"]),
                tx=float(row["tx_cost"]),
                tt=float(row.get("timing", {}).get("total_time_sec", 0.0)),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _log_result_to_tensorboard(writer: "SummaryWriter", result: Dict[str, object], step: int) -> None:
    """
    将单个候选架构的核心指标写入 TensorBoard。
    """
    writer.add_scalar("search/score", float(result["score"]), step)
    writer.add_scalar("search/mean_acc", float(result["mean_acc"]), step)
    writer.add_scalar("search/worst_acc", float(result["worst_acc"]), step)
    writer.add_scalar("search/robust_gap", float(result["robust_gap"]), step)
    writer.add_scalar("search/mean_cr", float(result["mean_cr"]), step)
    writer.add_scalar("search/std_cr", float(result["std_cr"]), step)
    writer.add_scalar("search/param_m", float(result["param_m"]), step)
    writer.add_scalar("search/tx_cost", float(result["tx_cost"]), step)

    score_breakdown = result.get("score_breakdown", {})
    for key, value in score_breakdown.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"score_breakdown/{key}", float(value), step)

    timing = result.get("timing", {})
    for key, value in timing.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"timing/{key}", float(value), step)

    resource = result.get("resource", {})
    peak_mem = resource.get("peak_gpu_mem_mb")
    if isinstance(peak_mem, (int, float)):
        writer.add_scalar("resource/peak_gpu_mem_mb", float(peak_mem), step)

    arch = result["arch"]
    writer.add_scalar("arch/insertion_stage", float(arch["insertion_stage"]), step)
    writer.add_scalar("arch/se_ratio", float(arch["se_ratio"]), step)
    writer.add_scalar("arch/cr", float(arch["cr"]), step)
    writer.add_scalar("arch/bottleneck_channels", float(arch["bottleneck_channels"]), step)
    writer.add_scalar("arch/ae_depth", float(arch["ae_depth"]), step)
    writer.add_scalar("arch/kernel_size", float(arch["kernel_size"]), step)
    writer.add_scalar("arch/use_skip", float(1 if arch["use_skip"] else 0), step)

    for channel_type, acc in result.get("per_channel_mean_acc", {}).items():
        writer.add_scalar(f"eval/channel_acc/{channel_type}", float(acc), step)
    for snr_tag, acc in result.get("per_snr_mean_acc", {}).items():
        writer.add_scalar(f"eval/snr_acc/{snr_tag}", float(acc), step)

    quick_train = result.get("quick_train", {})
    for epoch_row in quick_train.get("epochs", []):
        epoch = int(epoch_row.get("epoch", 0))
        epoch_step = (step - 1) * max(1, len(quick_train.get("epochs", []))) + epoch
        writer.add_scalar("quick_train/loss", float(epoch_row.get("train_loss", 0.0)), epoch_step)
        writer.add_scalar("quick_train/acc", float(epoch_row.get("train_acc", 0.0)), epoch_step)
        writer.add_scalar("quick_train/mean_cr", float(epoch_row.get("train_mean_cr", 0.0)), epoch_step)
        writer.add_scalar("quick_train/std_cr", float(epoch_row.get("train_std_cr", 0.0)), epoch_step)
        writer.add_scalar("quick_train/epoch_time_sec", float(epoch_row.get("epoch_time_sec", 0.0)), epoch_step)


def parse_args() -> argparse.Namespace:
    """搜索脚本参数定义。"""
    parser = argparse.ArgumentParser(description="Channel-aware multi-objective architecture search.")
    parser.add_argument("--dataset_name", type=str, default="UCMerced_LandUse")
    parser.add_argument("--train_dir", type=str, default="data/UCMerced_LandUse/UCMerced_LandUse-train")
    parser.add_argument("--valid_dir", type=str, default="data/UCMerced_LandUse/UCMerced_LandUse-valid")
    parser.add_argument("--output_dir", type=str, default="runs/nas_search")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--search_epochs", type=int, default=2)
    parser.add_argument("--search_lr", type=float, default=3e-4)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)

    parser.add_argument("--eval_channel_types", type=str, default="AWGN,Fading,Combined_channel")
    parser.add_argument("--eval_snr_list", type=str, default="0,4,8,12,16,20,24,28")

    parser.add_argument("--lambda_param", type=float, default=0.01)
    parser.add_argument("--lambda_tx", type=float, default=0.01)
    parser.add_argument("--lambda_robust", type=float, default=0.2)
    parser.add_argument("--lambda_cr", type=float, default=0.2)
    parser.add_argument("--target_cr", type=float, default=0.7)
    # Backward-compatible aliases. Prefer --lambda_cr / --target_cr in docs and commands.
    parser.add_argument("--lambda_rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target_rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--disable_dynamic_rate", action="store_true")
    parser.add_argument("--disable_channel_condition", action="store_true")
    parser.add_argument("--min_dynamic_cr", type=float, default=0.3)
    parser.add_argument("--max_dynamic_cr", type=float, default=1.0)
    parser.add_argument("--rate_blend_alpha", type=float, default=0.7)
    parser.add_argument("--disable_pretrained_backbone", action="store_true")
    parser.add_argument("--disable_tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    搜索入口：
    - 采样 num_samples 个候选架构
    - 逐个评估并写入 JSONL
    - 导出最佳架构与 top-k 汇总
    """
    args = parse_args()
    if args.lambda_rate is not None:
        args.lambda_cr = float(args.lambda_rate)
    if args.target_rate is not None:
        args.target_cr = float(args.target_rate)
    set_seed(args.seed)
    run_start = time.time()

    train_dir = resolve_path(args.train_dir)
    valid_dir = resolve_path(args.valid_dir)
    output_dir = resolve_path(args.output_dir) / f"{args.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"Train dir does not exist: {train_dir}")
    if not valid_dir.exists():
        raise FileNotFoundError(f"Valid dir does not exist: {valid_dir}")

    train_loader, valid_loader, class_names = build_dataloaders(
        train_dir=train_dir,
        valid_dir=valid_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    num_classes = len(class_names)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Search output dir: {output_dir}")
    print(f"Device: {requested_device}")
    print(f"Num classes: {num_classes}")

    search_space = SearchSpace.default()
    rng = random.Random(args.seed)
    results_path = output_dir / "search_results.jsonl"
    progress_path = output_dir / "search_progress.jsonl"
    tensorboard_dir = output_dir / "tensorboard"

    writer = None
    if not args.disable_tensorboard:
        if SummaryWriter is None:
            print("Warning: tensorboard is not available, skip SummaryWriter logging.")
        else:
            writer = SummaryWriter(str(tensorboard_dir))

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "paths": {
            "train_dir": str(train_dir),
            "valid_dir": str(valid_dir),
            "output_dir": str(output_dir),
        },
        "runtime": {
            "device": str(requested_device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "num_classes": num_classes,
            "class_names": list(class_names),
            "train_size": len(train_loader.dataset),
            "valid_size": len(valid_loader.dataset),
        },
        "search_space": {
            "total_candidates": _search_space_size(search_space),
            "insertion_stage": search_space.insertion_stage,
            "se_ratio": search_space.se_ratio,
            "cr": search_space.cr,
            "bottleneck_channels": search_space.bottleneck_channels,
            "ae_depth": search_space.ae_depth,
            "kernel_size": search_space.kernel_size,
            "use_skip": search_space.use_skip,
        },
        "args": vars(args),
        "tensorboard": {
            "enabled": bool(writer is not None),
            "log_dir": str(tensorboard_dir),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    all_results: List[Dict[str, object]] = []
    best_result: Dict[str, object] | None = None

    for idx in range(args.num_samples):
        arch = search_space.sample(rng)
        local_seed = args.seed + idx
        print(f"[{idx + 1}/{args.num_samples}] Searching arch: {arch.tag}")

        result = search_once(
            arch=arch,
            num_classes=num_classes,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=requested_device,
            args=args,
            local_seed=local_seed,
        )
        all_results.append(result)
        # 逐条追加写入，避免中途异常导致所有结果丢失。
        append_jsonl(results_path, result)
        append_jsonl(
            progress_path,
            {
                "index": idx + 1,
                "num_samples": args.num_samples,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "arch_tag": result["arch_tag"],
                "score": result["score"],
                "mean_acc": result["mean_acc"],
                "mean_cr": result["mean_cr"],
                "total_time_sec": result.get("timing", {}).get("total_time_sec", 0.0),
                "elapsed_run_sec": float(time.time() - run_start),
            },
        )
        if writer is not None:
            _log_result_to_tensorboard(writer, result, step=idx + 1)
        print(
            f"  score={result['score']:.4f}, mean_acc={result['mean_acc']:.4f}, "
            f"mean_cr={result['mean_cr']:.3f}, params(M)={result['param_m']:.3f}, tx_cost={result['tx_cost']:.1f}"
        )

        if best_result is None or float(result["score"]) > float(best_result["score"]):
            best_result = result

    if best_result is None:
        raise RuntimeError("No architectures were evaluated.")

    ranked = sorted(all_results, key=lambda row: float(row["score"]), reverse=True)
    top_k = ranked[: max(1, args.top_k)]

    best_arch = ArchitectureConfig.from_dict(best_result["arch"])
    save_architecture(output_dir / "best_arch.json", best_arch)

    summary = {
        "dataset_name": args.dataset_name,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "search_time_sec": float(time.time() - run_start),
        "results_files": {
            "search_results_jsonl": str(results_path),
            "search_progress_jsonl": str(progress_path),
            "best_arch_json": str(output_dir / "best_arch.json"),
            "topk_arches_json": str(output_dir / "topk_arches.json"),
            "ranked_results_json": str(output_dir / "ranked_results.json"),
            "ranked_results_csv": str(output_dir / "ranked_results.csv"),
            "topk_summary_md": str(output_dir / "topk_summary.md"),
            "run_config_json": str(output_dir / "run_config.json"),
        },
        "dynamic_selector": {
            "use_dynamic_rate": not args.disable_dynamic_rate,
            "use_channel_condition": not args.disable_channel_condition,
            "min_dynamic_cr": args.min_dynamic_cr,
            "max_dynamic_cr": args.max_dynamic_cr,
            "rate_blend_alpha": args.rate_blend_alpha,
            "target_cr": args.target_cr,
            "lambda_cr": args.lambda_cr,
        },
        "best_result": best_result,
        "top_k": top_k,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "ranked_results.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    (output_dir / "topk_arches.json").write_text(
        json.dumps([row["arch"] for row in top_k], indent=2), encoding="utf-8"
    )
    _write_ranked_csv(output_dir / "ranked_results.csv", ranked)
    _write_topk_markdown(output_dir / "topk_summary.md", top_k)
    if writer is not None:
        writer.add_text("best/arch_tag", str(best_result["arch_tag"]))
        writer.add_text("best/arch_json", json.dumps(best_result["arch"], ensure_ascii=False))
        writer.add_scalar("best/score", float(best_result["score"]), 0)
        writer.add_scalar("best/mean_acc", float(best_result["mean_acc"]), 0)
        writer.add_scalar("best/worst_acc", float(best_result["worst_acc"]), 0)
        writer.add_scalar("best/mean_cr", float(best_result["mean_cr"]), 0)
        writer.close()
    print(f"Done. Best architecture saved to: {output_dir / 'best_arch.json'}")


if __name__ == "__main__":
    main()
