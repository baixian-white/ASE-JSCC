from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.nas.nas_utils import resolve_path  # noqa: E402


def dominates(a: Dict[str, object], b: Dict[str, object]) -> bool:
    """
    Multi-objective dominance relation:
    - maximize: mean_acc
    - minimize: param_m, tx_cost, robust_gap
    """
    a_mean_acc = float(a["mean_acc"])
    b_mean_acc = float(b["mean_acc"])
    a_param = float(a["param_m"])
    b_param = float(b["param_m"])
    a_tx = float(a["tx_cost"])
    b_tx = float(b["tx_cost"])
    a_robust = float(a["robust_gap"])
    b_robust = float(b["robust_gap"])

    no_worse = (
        a_mean_acc >= b_mean_acc
        and a_param <= b_param
        and a_tx <= b_tx
        and a_robust <= b_robust
    )
    strictly_better = (
        a_mean_acc > b_mean_acc
        or a_param < b_param
        or a_tx < b_tx
        or a_robust < b_robust
    )
    return no_worse and strictly_better


def pareto_front(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    front: List[Dict[str, object]] = []
    for candidate in rows:
        is_dominated = False
        to_remove: List[int] = []
        for i, current in enumerate(front):
            if dominates(current, candidate):
                is_dominated = True
                break
            if dominates(candidate, current):
                to_remove.append(i)
        if not is_dominated:
            for idx in reversed(to_remove):
                front.pop(idx)
            front.append(candidate)
    return front


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Pareto front from NAS search results.")
    parser.add_argument(
        "--results_jsonl",
        type=str,
        required=True,
        help="Path to search_results.jsonl from search_channel_aware.py",
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results_jsonl)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("search_results.jsonl is empty.")

    front = pareto_front(rows)
    front_sorted = sorted(front, key=lambda row: float(row["mean_acc"]), reverse=True)
    top_k = front_sorted[: max(1, args.top_k)]

    if args.output_dir.strip():
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "pareto_front.json").write_text(json.dumps(front_sorted, indent=2), encoding="utf-8")
    (output_dir / "pareto_topk.json").write_text(json.dumps(top_k, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Pareto Front Summary",
        "",
        f"Total searched architectures: {len(rows)}",
        f"Pareto front size: {len(front_sorted)}",
        "",
        "| rank | arch_tag | mean_acc | worst_acc | mean_cr | param_m | tx_cost | robust_gap | score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top_k, start=1):
        markdown_lines.append(
            "| {rank} | {tag} | {mean:.4f} | {worst:.4f} | {mcr:.4f} | {pm:.3f} | {tx:.1f} | {rg:.4f} | {score:.4f} |".format(
                rank=idx,
                tag=row["arch_tag"],
                mean=float(row["mean_acc"]),
                worst=float(row["worst_acc"]),
                mcr=float(row.get("mean_cr", 0.0)),
                pm=float(row["param_m"]),
                tx=float(row["tx_cost"]),
                rg=float(row["robust_gap"]),
                score=float(row["score"]),
            )
        )
    (output_dir / "pareto_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Pareto outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
