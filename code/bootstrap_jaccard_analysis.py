#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3: Bootstrap Confidence Intervals for Jaccard Overlap

从现有 AtP 结果中 bootstrap resample，为所有 pairwise Jaccard overlap 计算 95% CI。
纯 CPU 计算，无需 GPU。

用法:
    python bootstrap_jaccard_analysis.py
    python bootstrap_jaccard_analysis.py --top-k 20 --n-bootstrap 1000
    python bootstrap_jaccard_analysis.py --results-dir output/qwen3-8b
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from itertools import combinations

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
os.chdir(PROJECT_ROOT)

from cross_domain_head_analysis import (
    find_latest_result_file, load_results, extract_patching_matrix,
    identify_important_heads, heads_to_set
)

ALL_DATASETS = [
    "chinese_legal", "english_legal", "chinese_moral", "english_moral",
    "general_mcqa"
]

DATASET_LABELS = {
    "chinese_legal": "Chinese Legal",
    "english_legal": "English Legal",
    "chinese_moral": "Chinese Moral",
    "english_moral": "English Moral",
    "general_mcqa": "General MCQA (STEM)",
    "general_mcqa_humanities": "General MCQA (Humanities)",
}


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def find_atp_result_file(results_dir: Path, dataset_key: str) -> Optional[Path]:
    """找到指定数据集的最新 AtP 结果文件"""
    pattern = f"{dataset_key}_atp_results_*.json"
    files = list(results_dir.glob(pattern))
    if not files:
        return find_latest_result_file(results_dir, dataset_key)
    return max(files, key=lambda x: x.stat().st_size)


def compute_jaccard(set_a: Set, set_b: Set) -> float:
    """计算 Jaccard 相似度"""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def bootstrap_jaccard(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    top_k: int = 20,
    n_bootstrap: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Bootstrap resample 两个数据集的 patching 矩阵，计算 Jaccard CI

    Args:
        matrix_a: (n_samples_a, n_layers, n_heads)
        matrix_b: (n_samples_b, n_layers, n_heads)
        top_k: 取 top-K heads
        n_bootstrap: bootstrap 迭代次数
        random_seed: 随机种子

    Returns:
        Dict with point estimate, CI, and bootstrap distribution
    """
    rng = np.random.RandomState(random_seed)
    n_a = matrix_a.shape[0]
    n_b = matrix_b.shape[0]

    jaccard_samples = []

    for i in range(n_bootstrap):
        # Resample with replacement
        idx_a = rng.choice(n_a, size=n_a, replace=True)
        idx_b = rng.choice(n_b, size=n_b, replace=True)

        resampled_a = matrix_a[idx_a]
        resampled_b = matrix_b[idx_b]

        # Identify top-K heads from resampled data
        heads_a = identify_important_heads(resampled_a, top_k=top_k)
        heads_b = identify_important_heads(resampled_b, top_k=top_k)

        set_a = heads_to_set(heads_a)
        set_b = heads_to_set(heads_b)

        jaccard = compute_jaccard(set_a, set_b)
        jaccard_samples.append(jaccard)

    jaccard_samples = np.array(jaccard_samples)

    # Point estimate from full data
    full_heads_a = identify_important_heads(matrix_a, top_k=top_k)
    full_heads_b = identify_important_heads(matrix_b, top_k=top_k)
    point_estimate = compute_jaccard(heads_to_set(full_heads_a), heads_to_set(full_heads_b))

    # 95% CI (percentile method)
    ci_lower = float(np.percentile(jaccard_samples, 2.5))
    ci_upper = float(np.percentile(jaccard_samples, 97.5))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "mean": float(np.mean(jaccard_samples)),
        "std": float(np.std(jaccard_samples)),
        "median": float(np.median(jaccard_samples)),
        "n_bootstrap": n_bootstrap,
        "bootstrap_distribution": jaccard_samples.tolist(),
    }


def run_bootstrap_analysis(
    results_dir: Path,
    output_dir: Path,
    top_k: int = 20,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    include_humanities: bool = False,
):
    """运行完整的 bootstrap 分析"""
    log("=" * 70)
    log("A3: Bootstrap Jaccard Confidence Intervals")
    log("=" * 70)
    log(f"Top-K: {top_k}, Bootstrap iterations: {n_bootstrap}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有数据集的 patching 矩阵
    datasets = ALL_DATASETS.copy()
    if include_humanities:
        datasets.append("general_mcqa_humanities")

    matrices = {}
    for ds in datasets:
        result_file = find_atp_result_file(results_dir, ds)
        if result_file is None:
            log(f"  {ds}: 未找到结果文件，跳过")
            continue

        raw = load_results(result_file)
        sample_results = raw.get("results", [])
        matrix = extract_patching_matrix(sample_results)
        if matrix is None:
            log(f"  {ds}: 无有效矩阵，跳过")
            continue

        matrices[ds] = matrix
        log(f"  {ds}: {matrix.shape[0]} samples, matrix shape {matrix.shape}")

    if len(matrices) < 2:
        log("数据不足（至少需要2个数据集）", )
        return None

    # 2. 计算所有 pairwise Jaccard CIs
    log(f"\n计算 {len(matrices)}C2 = {len(matrices)*(len(matrices)-1)//2} 对 Jaccard CIs...")

    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k,
            "n_bootstrap": n_bootstrap,
            "random_seed": random_seed,
            "datasets": list(matrices.keys()),
            "dataset_sizes": {ds: int(m.shape[0]) for ds, m in matrices.items()},
        },
        "pairwise": {},
        "matrix": {},
    }

    ds_list = sorted(matrices.keys())
    pair_count = 0

    for ds_a, ds_b in combinations(ds_list, 2):
        pair_count += 1
        pair_key = f"{ds_a}_vs_{ds_b}"
        log(f"  [{pair_count}] {DATASET_LABELS.get(ds_a, ds_a)} vs {DATASET_LABELS.get(ds_b, ds_b)}...")

        boot_result = bootstrap_jaccard(
            matrices[ds_a], matrices[ds_b],
            top_k=top_k,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + pair_count
        )

        # Store without full distribution (too large for JSON)
        result_summary = {k: v for k, v in boot_result.items() if k != "bootstrap_distribution"}
        result_summary["dataset_a"] = ds_a
        result_summary["dataset_b"] = ds_b
        result_summary["label_a"] = DATASET_LABELS.get(ds_a, ds_a)
        result_summary["label_b"] = DATASET_LABELS.get(ds_b, ds_b)

        results["pairwise"][pair_key] = result_summary

        log(f"    Jaccard = {boot_result['point_estimate']:.3f} "
            f"[{boot_result['ci_lower']:.3f}, {boot_result['ci_upper']:.3f}]")

    # 3. 构建 Jaccard 矩阵 (for easy table formatting)
    for ds_a in ds_list:
        results["matrix"][ds_a] = {}
        for ds_b in ds_list:
            if ds_a == ds_b:
                results["matrix"][ds_a][ds_b] = {
                    "point_estimate": 1.0, "ci_lower": 1.0, "ci_upper": 1.0
                }
            else:
                pair_key = f"{min(ds_a,ds_b)}_vs_{max(ds_a,ds_b)}"
                if pair_key not in results["pairwise"]:
                    pair_key = f"{max(ds_a,ds_b)}_vs_{min(ds_a,ds_b)}"
                if pair_key in results["pairwise"]:
                    p = results["pairwise"][pair_key]
                    results["matrix"][ds_a][ds_b] = {
                        "point_estimate": p["point_estimate"],
                        "ci_lower": p["ci_lower"],
                        "ci_upper": p["ci_upper"],
                    }

    # 4. 分类分析：normative-normative vs normative-general
    normative_ds = [ds for ds in ds_list if ds in ["chinese_legal", "english_legal", "chinese_moral", "english_moral"]]
    general_ds = [ds for ds in ds_list if ds.startswith("general_")]

    norm_norm_jaccards = []
    norm_gen_jaccards = []

    for pair_key, pair_data in results["pairwise"].items():
        ds_a = pair_data["dataset_a"]
        ds_b = pair_data["dataset_b"]
        j = pair_data["point_estimate"]

        if ds_a in normative_ds and ds_b in normative_ds:
            norm_norm_jaccards.append(j)
        elif (ds_a in normative_ds and ds_b in general_ds) or \
             (ds_a in general_ds and ds_b in normative_ds):
            norm_gen_jaccards.append(j)

    results["category_analysis"] = {
        "normative_normative": {
            "mean": float(np.mean(norm_norm_jaccards)) if norm_norm_jaccards else None,
            "std": float(np.std(norm_norm_jaccards)) if norm_norm_jaccards else None,
            "values": norm_norm_jaccards,
            "n_pairs": len(norm_norm_jaccards),
        },
        "normative_general": {
            "mean": float(np.mean(norm_gen_jaccards)) if norm_gen_jaccards else None,
            "std": float(np.std(norm_gen_jaccards)) if norm_gen_jaccards else None,
            "values": norm_gen_jaccards,
            "n_pairs": len(norm_gen_jaccards),
        },
    }

    # 5. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"bootstrap_jaccard_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"\nJSON 结果已保存: {json_path}")

    # 6. 生成报告
    report = generate_report(results)
    report_path = output_dir / f"bootstrap_jaccard_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"报告已保存: {report_path}")
    print("\n" + report)

    return results


def generate_report(results: Dict) -> str:
    """生成文本报告"""
    lines = [
        "=" * 70,
        "A3: Bootstrap Jaccard Confidence Intervals Report",
        "=" * 70,
        f"Generated: {results['metadata']['timestamp']}",
        f"Top-K: {results['metadata']['top_k']}",
        f"Bootstrap iterations: {results['metadata']['n_bootstrap']}",
        "",
    ]

    # Pairwise results table
    lines.append("Pairwise Jaccard Overlap with 95% Bootstrap CIs:")
    lines.append("-" * 70)
    lines.append(f"{'Pair':<45} {'Jaccard':>8} {'95% CI':>20}")
    lines.append("-" * 70)

    for pair_key, pair_data in sorted(results["pairwise"].items()):
        label = f"{pair_data['label_a']} vs {pair_data['label_b']}"
        j = pair_data["point_estimate"]
        ci = f"[{pair_data['ci_lower']:.3f}, {pair_data['ci_upper']:.3f}]"
        lines.append(f"{label:<45} {j:>8.3f} {ci:>20}")

    lines.append("")

    # Jaccard matrix with CIs (LaTeX-friendly)
    lines.append("Jaccard Matrix (for Table 3):")
    lines.append("-" * 70)
    ds_list = sorted(results["metadata"]["datasets"])
    header = f"{'':>20}" + "".join(f"{DATASET_LABELS.get(ds,ds):>18}" for ds in ds_list)
    lines.append(header)

    for ds_a in ds_list:
        row = f"{DATASET_LABELS.get(ds_a,ds_a):>20}"
        for ds_b in ds_list:
            cell = results["matrix"].get(ds_a, {}).get(ds_b, {})
            if ds_a == ds_b:
                row += f"{'1.000':>18}"
            elif cell:
                j = cell["point_estimate"]
                ci_l = cell["ci_lower"]
                ci_u = cell["ci_upper"]
                row += f"  {j:.3f}[{ci_l:.2f},{ci_u:.2f}]"
            else:
                row += f"{'N/A':>18}"
        lines.append(row)

    lines.append("")

    # Category analysis
    cat = results.get("category_analysis", {})
    nn = cat.get("normative_normative", {})
    ng = cat.get("normative_general", {})

    lines.append("Category Analysis:")
    lines.append("-" * 70)
    if nn.get("mean") is not None:
        lines.append(f"  Normative-Normative mean Jaccard: {nn['mean']:.3f} +/- {nn['std']:.3f} "
                     f"(n={nn['n_pairs']} pairs)")
    if ng.get("mean") is not None:
        lines.append(f"  Normative-General mean Jaccard:   {ng['mean']:.3f} +/- {ng['std']:.3f} "
                     f"(n={ng['n_pairs']} pairs)")

    if nn.get("mean") is not None and ng.get("mean") is not None:
        diff = nn["mean"] - ng["mean"]
        lines.append(f"  Difference (norm-norm - norm-gen): {diff:+.3f}")
        if diff > 0:
            lines.append("  -> Normative datasets share more heads with each other than with general MCQA")
        else:
            lines.append("  -> No clear separation between normative-normative and normative-general overlap")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="A3: Bootstrap Jaccard Confidence Intervals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--results-dir", type=str, default="output/qwen3-8b",
                        help="AtP 结果目录")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K heads")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap 迭代次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--include-humanities", action="store_true",
                        help="包含 Humanities 数据集")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "bootstrap"

    run_bootstrap_analysis(
        results_dir=results_dir,
        output_dir=output_dir,
        top_k=args.top_k,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
        include_humanities=args.include_humanities,
    )


if __name__ == "__main__":
    main()
