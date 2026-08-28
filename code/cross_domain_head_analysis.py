

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

def find_latest_result_file(results_dir: Path, dataset_key: str) -> Optional[Path]:
    pattern = f"{dataset_key}_*_results_*.json"
    files = list(results_dir.glob(pattern))

    if not files:
        return None

    return max(files, key=lambda x: x.stat().st_mtime)

def load_results(results_file: Path) -> Dict:
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_patching_matrix(results: List[Dict]) -> np.ndarray:

    valid_results = [r for r in results if r.get("patching_result") is not None]

    if not valid_results:
        return None

    first_result = valid_results[0]["patching_result"]
    n_layers = len(first_result)
    n_heads = len(first_result["layer_0"])
    n_samples = len(valid_results)

    matrix = np.zeros((n_samples, n_layers, n_heads))

    for sample_idx, result in enumerate(valid_results):
        patching = result["patching_result"]
        for layer_idx in range(n_layers):
            layer_key = f"layer_{layer_idx}"
            for head_idx in range(n_heads):
                head_key = f"head_{head_idx}"
                matrix[sample_idx, layer_idx, head_idx] = patching[layer_key][head_key]["prob_diff"]

    return matrix

def identify_important_heads(
    matrix: np.ndarray,
    method: str = "mean_abs",
    top_k: int = 20,
    threshold: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    n_layers = matrix.shape[1]
    n_heads = matrix.shape[2]

    if method == "mean_abs":
        scores = np.mean(np.abs(matrix), axis=0)
    elif method == "max_abs":
        scores = np.max(np.abs(matrix), axis=0)
    elif method == "mean":
        scores = np.mean(matrix, axis=0)
    elif method == "std":
        scores = np.std(matrix, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    heads = []
    for layer in range(n_layers):
        for head in range(n_heads):
            heads.append((layer, head, scores[layer, head]))

    heads.sort(key=lambda x: abs(x[2]), reverse=True)

    if threshold is not None:
        heads = [(l, h, s) for l, h, s in heads if abs(s) >= threshold]
    else:
        heads = heads[:top_k]

    return heads

def heads_to_set(heads: List[Tuple[int, int, float]]) -> Set[Tuple[int, int]]:
    return {(layer, head) for layer, head, _ in heads}

def compute_overlap(heads1: Set[Tuple[int, int]], heads2: Set[Tuple[int, int]]) -> Dict:
    intersection = heads1 & heads2
    union = heads1 | heads2

    jaccard = len(intersection) / len(union) if union else 0

    return {
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
        "overlap_count": len(intersection),
        "total_unique": len(union),
        "only_in_first": heads1 - heads2,
        "only_in_second": heads2 - heads1,
    }

def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    method: str = "mean_abs",
    colorscale: str = "Viridis",
) -> go.Figure:

    if method == "mean_abs":
        aggregated = np.mean(np.abs(matrix), axis=0)
    elif method == "max_abs":
        aggregated = np.max(np.abs(matrix), axis=0)
    elif method == "mean":
        aggregated = np.mean(matrix, axis=0)
    else:
        aggregated = np.mean(np.abs(matrix), axis=0)

    n_layers, n_heads = aggregated.shape

    fig = px.imshow(
        aggregated,
        x=[f"H{i}" for i in range(n_heads)],
        y=[f"L{i}" for i in range(n_layers)],
        color_continuous_scale=colorscale,
        labels=dict(x="Head", y="Layer", color="Score"),
        title=title,
    )

    fig.update_layout(
        width=800,
        height=1000,
        xaxis=dict(side="top"),
    )

    return fig

def plot_overlap_comparison(
    heads_dict: Dict[str, Set[Tuple[int, int]]],
    title: str = "Head Overlap Analysis",
) -> go.Figure:
    datasets = list(heads_dict.keys())
    n = len(datasets)

    overlap_matrix = np.zeros((n, n))

    for i, name1 in enumerate(datasets):
        for j, name2 in enumerate(datasets):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                overlap = compute_overlap(heads_dict[name1], heads_dict[name2])
                overlap_matrix[i, j] = overlap["jaccard"]

    fig = px.imshow(
        overlap_matrix,
        x=datasets,
        y=datasets,
        color_continuous_scale="Blues",
        labels=dict(color="Jaccard"),
        title=title,
        text_auto=".2f",
    )

    fig.update_layout(
        width=600,
        height=500,
    )

    return fig

def plot_top_heads_bar(
    heads_dict: Dict[str, List[Tuple[int, int, float]]],
    top_k: int = 15,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(heads_dict.keys()),
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (name, heads) in enumerate(heads_dict.items()):
        row, col = positions[idx]

        heads_top = heads[:top_k]
        labels = [f"L{l}-H{h}" for l, h, _ in heads_top]
        scores = [abs(s) for _, _, s in heads_top]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                name=name,
                marker_color=px.colors.qualitative.Plotly[idx],
            ),
            row=row, col=col,
        )

    fig.update_layout(
        title="Top Attention Heads by Dataset",
        height=800,
        showlegend=False,
    )

    return fig

def plot_venn_style_overlap(
    legal_heads: Set[Tuple[int, int]],
    moral_heads: Set[Tuple[int, int]],
    chinese_heads: Set[Tuple[int, int]],
    english_heads: Set[Tuple[int, int]],
) -> go.Figure:

    overlaps = {
        "Legal vs Moral": compute_overlap(legal_heads, moral_heads),
        "Chinese vs English": compute_overlap(chinese_heads, english_heads),
        "Chinese Legal vs English Legal": None,
        "Chinese Moral vs English Moral": None,
    }

    fig = go.Figure()

    categories = ["Legal∩Moral", "Chinese∩English"]
    values = [
        overlaps["Legal vs Moral"]["jaccard"],
        overlaps["Chinese vs English"]["jaccard"],
    ]

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{v:.2%}" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        title="Cross-Domain Head Overlap (Jaccard Index)",
        yaxis_title="Jaccard Index",
        yaxis_range=[0, 1],
        height=400,
    )

    return fig

def generate_analysis_report(
    results_dict: Dict[str, Dict],
    heads_dict: Dict[str, List[Tuple[int, int, float]]],
    output_dir: Path,
) -> str:
    report_lines = [
        "# Cross-Domain Head Analysis Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\nModel: Qwen3-8B",
        "\n---\n",
    ]

    report_lines.append("## 1. Dataset Statistics\n")
    for name, results in results_dict.items():
        meta = results.get("metadata", {})
        n_results = len(results.get("results", []))
        report_lines.append(f"- **{name}**: {n_results} samples")

    report_lines.append("\n## 2. Top-20 Important Attention Heads\n")
    for name, heads in heads_dict.items():
        report_lines.append(f"\n### {name}\n")
        report_lines.append("| Rank | Layer | Head | Score |")
        report_lines.append("|------|-------|------|-------|")
        for i, (layer, head, score) in enumerate(heads[:20], 1):
            report_lines.append(f"| {i} | {layer} | {head} | {score:.6f} |")

    report_lines.append("\n## 3. Cross-Domain Overlap Analysis\n")

    legal_heads = set()
    moral_heads = set()
    chinese_heads = set()
    english_heads = set()

    for name, heads in heads_dict.items():
        head_set = heads_to_set(heads)
        if "legal" in name.lower():
            legal_heads |= head_set
        if "moral" in name.lower():
            moral_heads |= head_set
        if "chinese" in name.lower():
            chinese_heads |= head_set
        if "english" in name.lower():
            english_heads |= head_set

    overlap_legal_moral = compute_overlap(legal_heads, moral_heads)
    overlap_zh_en = compute_overlap(chinese_heads, english_heads)

    report_lines.append("### Legal vs Moral\n")
    report_lines.append(f"- Jaccard coefficient: {overlap_legal_moral['jaccard']:.4f}")
    report_lines.append(f"- Shared head count: {overlap_legal_moral['overlap_count']}")
    report_lines.append(f"- Legal-only heads: {len(overlap_legal_moral['only_in_first'])}")
    report_lines.append(f"- Moral-only heads: {len(overlap_legal_moral['only_in_second'])}")

    report_lines.append("\n### Chinese vs English\n")
    report_lines.append(f"- Jaccard coefficient: {overlap_zh_en['jaccard']:.4f}")
    report_lines.append(f"- Shared head count: {overlap_zh_en['overlap_count']}")
    report_lines.append(f"- Chinese-only heads: {len(overlap_zh_en['only_in_first'])}")
    report_lines.append(f"- English-only heads: {len(overlap_zh_en['only_in_second'])}")

    if overlap_legal_moral['intersection']:
        report_lines.append("\n### Heads shared by Legal and Moral\n")
        for layer, head in sorted(overlap_legal_moral['intersection']):
            report_lines.append(f"- L{layer}-H{head}")

    report_text = "\n".join(report_lines)

    report_file = output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report saved: {report_file}")
    return report_text

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Domain Head Analysis")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="output/qwen3-8b",
        help="Results directory",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K important head count",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="mean_abs",
        choices=["mean_abs", "max_abs", "mean", "std"],
        help="Aggregation method",
    )

    parser.add_argument(
        "--save-html",
        action="store_true",
        default=True,
        help="Save HTML visualization",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    results_dir = PROJECT_ROOT / args.results_dir

    print("=" * 60)
    print("Cross-Domain Head Analysis")
    print(f"Results directory: {results_dir}")
    print("=" * 60)

    datasets = ["chinese_legal", "english_legal", "chinese_moral", "english_moral"]
    results_dict = {}
    matrices_dict = {}
    heads_dict = {}

    for dataset_key in datasets:
        result_file = find_latest_result_file(results_dir, dataset_key)

        if result_file is None:
            print(f"Warning: results for {dataset_key} not found")
            continue

        print(f"Loaded: {result_file.name}")
        results = load_results(result_file)
        results_dict[dataset_key] = results

        matrix = extract_patching_matrix(results.get("results", results))
        if matrix is not None:
            matrices_dict[dataset_key] = matrix

            heads = identify_important_heads(matrix, method=args.method, top_k=args.top_k)
            heads_dict[dataset_key] = heads

            print(f"  {dataset_key}: {matrix.shape[0]} samples, {matrix.shape[1]} layers, {matrix.shape[2]} heads")

    if not matrices_dict:
        print("Error: no usable result data")
        return

    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, matrix in matrices_dict.items():
        fig = plot_heatmap(matrix, f"{name} Path Patching Heatmap", method=args.method)
        if args.save_html:
            fig.write_html(output_dir / f"{name}_heatmap_{timestamp}.html")

    if len(heads_dict) > 1:
        fig_bar = plot_top_heads_bar(heads_dict, top_k=15)
        if args.save_html:
            fig_bar.write_html(output_dir / f"top_heads_comparison_{timestamp}.html")

    if len(heads_dict) >= 2:
        heads_sets = {k: heads_to_set(v) for k, v in heads_dict.items()}
        fig_overlap = plot_overlap_comparison(heads_sets, "Head Overlap Analysis")
        if args.save_html:
            fig_overlap.write_html(output_dir / f"overlap_analysis_{timestamp}.html")

    generate_analysis_report(results_dict, heads_dict, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
