from __future__ import annotations

"""
NAS 搜索结果绘图脚本。

功能：
1) 从 NAS 搜索输出目录读取 ranked_results.json / search_results.jsonl。
2) 自动生成汇报常用图：
   - tx_cost vs mean_acc 散点图（颜色=robust_gap）
   - Pareto 前沿图（二维：tx_cost vs mean_acc）
   - Top-K 综合得分条形图（含 mean_acc 折线）
   - 最优架构在不同信道/SNR下的准确率曲线（若有 condition_rows）
3) 生成 figures_manifest.md，方便汇报时快速引用。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nas.nas_utils import resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot NAS search result figures for reporting.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="",
        help="Path to one NAS search run dir (contains ranked_results.json/search_results.jsonl).",
    )
    parser.add_argument(
        "--results_json",
        type=str,
        default="",
        help="Path to ranked_results.json (optional, overrides run_dir).",
    )
    parser.add_argument(
        "--results_jsonl",
        type=str,
        default="",
        help="Path to search_results.jsonl (fallback when ranked_results.json not provided).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Figure output directory. Default: <run_dir>/figures",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def _load_results(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Path]:
    run_dir = resolve_path(args.run_dir) if args.run_dir else None
    if args.results_json:
        json_path = resolve_path(args.results_json)
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        base_dir = json_path.parent
    elif run_dir and (run_dir / "ranked_results.json").exists():
        json_path = run_dir / "ranked_results.json"
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        base_dir = run_dir
    else:
        if args.results_jsonl:
            jsonl_path = resolve_path(args.results_jsonl)
            base_dir = jsonl_path.parent
        elif run_dir and (run_dir / "search_results.jsonl").exists():
            jsonl_path = run_dir / "search_results.jsonl"
            base_dir = run_dir
        else:
            raise FileNotFoundError("No ranked_results.json or search_results.jsonl found.")

        rows = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)

    if not rows:
        raise RuntimeError("No NAS rows loaded. Input file appears empty.")
    return rows, base_dir


def _to_frame(rows: List[Dict[str, object]]) -> pd.DataFrame:
    payload = []
    for idx, row in enumerate(rows, start=1):
        arch = row.get("arch", {})
        payload.append(
            {
                "rank": idx,
                "arch_tag": row.get("arch_tag", ""),
                "score": float(row.get("score", 0.0)),
                "mean_acc": float(row.get("mean_acc", 0.0)),
                "worst_acc": float(row.get("worst_acc", 0.0)),
                "robust_gap": float(row.get("robust_gap", 0.0)),
                "mean_cr": float(row.get("mean_cr", 0.0)),
                "std_cr": float(row.get("std_cr", 0.0)),
                "param_m": float(row.get("param_m", 0.0)),
                "tx_cost": float(row.get("tx_cost", 0.0)),
                "insertion_stage": arch.get("insertion_stage", None),
                "se_ratio": arch.get("se_ratio", None),
                "cr": arch.get("cr", None),
                "bottleneck_channels": arch.get("bottleneck_channels", None),
                "ae_depth": arch.get("ae_depth", None),
                "kernel_size": arch.get("kernel_size", None),
                "use_skip": arch.get("use_skip", None),
            }
        )
    return pd.DataFrame(payload)


def _pareto_front_2d(df: pd.DataFrame) -> pd.DataFrame:
    """
    二维 Pareto：最小化 tx_cost，最大化 mean_acc。
    """
    ordered = df.sort_values("tx_cost", ascending=True).reset_index(drop=True)
    best_acc = -1.0
    keep = []
    for _, row in ordered.iterrows():
        if float(row["mean_acc"]) >= best_acc:
            keep.append(True)
            best_acc = float(row["mean_acc"])
        else:
            keep.append(False)
    return ordered.loc[keep].copy()


def plot_scatter(df: pd.DataFrame, out_path: Path, top_k: int, dpi: int) -> str:
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    sc = ax.scatter(
        df["tx_cost"],
        df["mean_acc"],
        c=df["robust_gap"],
        cmap="viridis_r",
        s=55,
        alpha=0.9,
        edgecolors="none",
    )
    top = df.nsmallest(top_k, "rank")
    ax.scatter(
        top["tx_cost"],
        top["mean_acc"],
        facecolors="none",
        edgecolors="red",
        s=130,
        linewidths=1.4,
        label=f"Top-{top_k}",
    )
    ax.set_xlabel("Tx Cost")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("NAS Candidates: Tx Cost vs Mean Accuracy")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Robust Gap (lower is better)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return f"- `{out_path.name}`: 候选架构散点图（x=tx_cost, y=mean_acc, 颜色=robust_gap）"


def plot_pareto(df: pd.DataFrame, out_path: Path, dpi: int) -> str:
    front = _pareto_front_2d(df)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.scatter(df["tx_cost"], df["mean_acc"], color="#9ea7b3", s=36, alpha=0.7, label="All Candidates")
    ax.plot(front["tx_cost"], front["mean_acc"], color="#d62728", marker="o", linewidth=1.8, label="Pareto Front")
    ax.set_xlabel("Tx Cost")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Pareto Front (2D: Tx Cost vs Mean Accuracy)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return f"- `{out_path.name}`: 二维 Pareto 前沿图（tx_cost 与 mean_acc 折中）"


def plot_topk(df: pd.DataFrame, out_path: Path, top_k: int, dpi: int) -> str:
    top = df.nsmallest(top_k, "rank").copy()
    top["label"] = top["rank"].astype(str) + ":" + top["arch_tag"].str.slice(0, 18)

    fig, ax1 = plt.subplots(figsize=(10.8, 5.8))
    ax1.bar(top["label"], top["score"], color="#4c78a8", alpha=0.85, label="Score")
    ax1.set_ylabel("Score")
    ax1.set_xlabel("Architecture (rank:tag)")
    ax1.tick_params(axis="x", rotation=35, labelsize=8)
    ax1.grid(alpha=0.2, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(top["label"], top["mean_acc"], color="#e45756", marker="o", linewidth=1.8, label="Mean Accuracy")
    ax2.set_ylabel("Mean Accuracy")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"Top-{top_k} Architectures: Score and Mean Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return f"- `{out_path.name}`: Top-{top_k} 架构的 score 与 mean_acc 对比图"


def plot_best_channel_curve(rows: List[Dict[str, object]], out_path: Path, dpi: int) -> str | None:
    best = rows[0]
    condition_rows = best.get("condition_rows", [])
    if not condition_rows:
        return None

    frame = pd.DataFrame(condition_rows)
    if frame.empty:
        return None
    frame["snr_db"] = frame["snr_db"].astype(int)
    frame = frame.sort_values(["channel_type", "snr_db"])

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for channel_type, part in frame.groupby("channel_type"):
        ax.plot(part["snr_db"], part["acc"], marker="o", linewidth=1.8, label=str(channel_type))

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Best Architecture Accuracy vs SNR ({best.get('arch_tag', 'best')})")
    ax.grid(alpha=0.25)
    ax.legend(title="Channel Type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return f"- `{out_path.name}`: 最优架构在不同信道/SNR 下的准确率曲线"


def write_manifest(path: Path, lines: List[str]) -> None:
    content = ["# NAS Figures Manifest", "", "生成图表如下：", "", *lines, ""]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, base_dir = _load_results(args)
    df = _to_frame(rows)

    output_dir = resolve_path(args.output_dir) if args.output_dir else (base_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    manifest_lines: List[str] = []
    manifest_lines.append(plot_scatter(df, output_dir / "fig1_scatter_tx_vs_acc.png", args.top_k, args.dpi))
    manifest_lines.append(plot_pareto(df, output_dir / "fig2_pareto_front.png", args.dpi))
    manifest_lines.append(plot_topk(df, output_dir / "fig3_topk_score_acc.png", args.top_k, args.dpi))
    best_curve_desc = plot_best_channel_curve(rows, output_dir / "fig4_best_arch_snr_curve.png", args.dpi)
    if best_curve_desc is not None:
        manifest_lines.append(best_curve_desc)

    write_manifest(output_dir / "figures_manifest.md", manifest_lines)
    print(f"Figures saved to: {output_dir}")
    for line in manifest_lines:
        print(line)


if __name__ == "__main__":
    main()
