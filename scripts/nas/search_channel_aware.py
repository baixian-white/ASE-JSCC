from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

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
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_str_list(values: str) -> List[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def evaluate_model(
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_channel_types: List[str],
    eval_snr_list: List[int],
    max_eval_batches: int,
) -> Dict[str, object]:
    model.eval()
    acc_map: Dict[str, float] = {}
    acc_values: List[float] = []
    cr_values: List[float] = []

    with torch.no_grad():
        for channel_type in eval_channel_types:
            for snr in eval_snr_list:
                correct = 0
                total = 0
                for batch_idx, (images, labels) in enumerate(valid_loader):
                    if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                        break
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images, channel_type=channel_type, snr_db=float(snr))
                    pred = torch.argmax(logits, dim=1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                    if hasattr(model, "last_forward_stats"):
                        cr_values.append(float(model.last_forward_stats.get("mean_cr", 0.0)))
                acc = correct / max(total, 1)
                key = f"{channel_type}@{snr}dB"
                acc_map[key] = acc
                acc_values.append(acc)

    mean_acc = sum(acc_values) / max(len(acc_values), 1)
    mean_cr = sum(cr_values) / max(len(cr_values), 1) if cr_values else 0.0
    std_cr = (
        float(torch.tensor(cr_values, dtype=torch.float32).std(unbiased=False).item())
        if len(cr_values) > 1
        else 0.0
    )
    return {
        "mean_acc": mean_acc,
        "worst_acc": min(acc_values) if acc_values else 0.0,
        "robust_gap": robust_gap(acc_values),
        "mean_cr": mean_cr,
        "std_cr": std_cr,
        "acc_map": acc_map,
    }


def quick_train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    search_epochs: int,
    learning_rate: float,
    max_train_batches: int,
    seed: int,
) -> None:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sampler = default_channel_sampler()
    rng = random.Random(seed)

    model.train()
    for _ in range(search_epochs):
        for batch_idx, (images, labels) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break
            channel_type, snr = sampler.sample(rng)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images, channel_type=channel_type, snr_db=float(snr))
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()


def score_architecture(
    mean_acc: float,
    param_m: float,
    tx_cost: float,
    robust_gap_val: float,
    mean_cr: float,
    lambda_param: float,
    lambda_tx: float,
    lambda_robust: float,
    lambda_rate: float,
    target_rate: float,
) -> float:
    # Normalize tx_cost to a similar scale as model params.
    tx_norm = tx_cost / 10_000.0
    rate_violation = max(0.0, mean_cr - target_rate)
    return (
        mean_acc
        - lambda_param * param_m
        - lambda_tx * tx_norm
        - lambda_robust * robust_gap_val
        - lambda_rate * rate_violation
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

    quick_train(
        model=model,
        train_loader=train_loader,
        device=device,
        search_epochs=args.search_epochs,
        learning_rate=args.search_lr,
        max_train_batches=args.max_train_batches,
        seed=local_seed,
    )

    eval_result = evaluate_model(
        model=model,
        valid_loader=valid_loader,
        device=device,
        eval_channel_types=parse_str_list(args.eval_channel_types),
        eval_snr_list=parse_int_list(args.eval_snr_list),
        max_eval_batches=args.max_eval_batches,
    )

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
        lambda_rate=args.lambda_rate,
        target_rate=args.target_rate,
    )

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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Channel-aware multi-objective architecture search.")
    parser.add_argument("--dataset_name", type=str, default="Soya")
    parser.add_argument("--train_dir", type=str, default="data/SoyaHealthVision/train")
    parser.add_argument("--valid_dir", type=str, default="data/SoyaHealthVision/valid")
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
    parser.add_argument("--lambda_rate", type=float, default=0.2)
    parser.add_argument("--target_rate", type=float, default=0.7)
    parser.add_argument("--disable_dynamic_rate", action="store_true")
    parser.add_argument("--disable_channel_condition", action="store_true")
    parser.add_argument("--min_dynamic_cr", type=float, default=0.3)
    parser.add_argument("--max_dynamic_cr", type=float, default=1.0)
    parser.add_argument("--rate_blend_alpha", type=float, default=0.7)
    parser.add_argument("--disable_pretrained_backbone", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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
        append_jsonl(results_path, result)
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
        "dynamic_selector": {
            "use_dynamic_rate": not args.disable_dynamic_rate,
            "use_channel_condition": not args.disable_channel_condition,
            "min_dynamic_cr": args.min_dynamic_cr,
            "max_dynamic_cr": args.max_dynamic_cr,
            "rate_blend_alpha": args.rate_blend_alpha,
            "target_rate": args.target_rate,
            "lambda_rate": args.lambda_rate,
        },
        "best_result": best_result,
        "top_k": top_k,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "topk_arches.json").write_text(
        json.dumps([row["arch"] for row in top_k], indent=2), encoding="utf-8"
    )
    print(f"Done. Best architecture saved to: {output_dir / 'best_arch.json'}")


if __name__ == "__main__":
    main()
