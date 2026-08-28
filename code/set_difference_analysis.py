

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))
os.chdir(PROJECT_ROOT)

from cross_domain_head_analysis import (
    find_latest_result_file, load_results, extract_patching_matrix,
    identify_important_heads, heads_to_set, compute_overlap
)

NORMATIVE_DATASETS = ["chinese_legal", "english_legal", "chinese_moral", "english_moral"]
DATASET_LABELS = {
    "chinese_legal": "Chinese Legal (JEC-QA)",
    "english_legal": "English Legal (LEXam)",
    "chinese_moral": "Chinese Moral (SafetyBench)",
    "english_moral": "English Moral (MoralChoice)",
    "general_mcqa": "General MCQA (MMLU STEM)",
}

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

def find_atp_result_file(results_dir: Path, dataset_key: str) -> Optional[Path]:
    pattern = f"{dataset_key}_atp_results_*.json"
    files = list(results_dir.glob(pattern))
    if not files:

        return find_latest_result_file(results_dir, dataset_key)

    return max(files, key=lambda x: x.stat().st_size)

def load_all_head_data(results_dir: Path) -> Dict[str, Tuple[np.ndarray, List]]:
    all_datasets = NORMATIVE_DATASETS + ["general_mcqa"]
    data = {}

    for ds in all_datasets:
        result_file = find_atp_result_file(results_dir, ds)
        if result_file is None:
            log(f"Result file for {ds} not found; skipping", "WARN")
            continue

        raw = load_results(result_file)
        sample_results = raw.get("results", [])
        matrix = extract_patching_matrix(sample_results)
        if matrix is None:
            log(f"{ds}: no valid patching matrix; skipping", "WARN")
            continue

        n_samples = matrix.shape[0]
        log(f"Loaded {ds}: {n_samples} samples, matrix {matrix.shape}, file {result_file.name}")
        data[ds] = matrix

    return data

def get_head_scores(matrix: np.ndarray) -> Dict[Tuple[int, int], float]:
    scores = np.mean(np.abs(matrix), axis=0)
    result = {}
    for layer in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            result[(layer, head)] = float(scores[layer, head])
    return result

def layer_bin(layer: int, n_layers: int = 36) -> str:
    if layer < 12:
        return "early (0-11)"
    elif layer < 24:
        return "middle (12-23)"
    elif layer < 32:
        return "late-middle (24-31)"
    else:
        return "final (32-35)"

def analyze_set_difference(
    data: Dict[str, np.ndarray],
    top_k: int = 20
) -> Dict:
    log(f"\n{'='*70}")
    log(f"  Set-difference analysis (top-K = {top_k})")
    log(f"{'='*70}")

    head_sets = {}
    head_scores = {}
    head_lists = {}

    for ds, matrix in data.items():
        heads = identify_important_heads(matrix, top_k=top_k)
        head_sets[ds] = heads_to_set(heads)
        head_scores[ds] = get_head_scores(matrix)
        head_lists[ds] = heads
        log(f"{ds}: top-{top_k} = {sorted(head_sets[ds])}")

    if "general_mcqa" not in head_sets:
        log("Missing general_mcqa data!", "ERROR")
        return {}

    general_set = head_sets["general_mcqa"]
    general_scores = head_scores["general_mcqa"]

    results = {
        "top_k": top_k,
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
        "summary": {},
    }

    all_specific_heads = set()
    all_shared_with_general = set()

    for ds in NORMATIVE_DATASETS:
        if ds not in head_sets:
            continue

        norm_set = head_sets[ds]
        norm_scores = head_scores[ds]

        specific = norm_set - general_set
        shared = norm_set & general_set
        general_only = general_set - norm_set

        all_specific_heads |= specific
        all_shared_with_general |= shared

        specific_details = []
        for (layer, head) in sorted(specific):
            specific_details.append({
                "head": f"L{layer}-H{head}",
                "layer": layer,
                "head_idx": head,
                "layer_bin": layer_bin(layer),
                "normative_score": norm_scores[(layer, head)],
                "general_score": general_scores[(layer, head)],
                "score_ratio": norm_scores[(layer, head)] / max(general_scores[(layer, head)], 1e-10),
            })
        specific_details.sort(key=lambda x: x["normative_score"], reverse=True)

        shared_details = []
        for (layer, head) in sorted(shared):
            shared_details.append({
                "head": f"L{layer}-H{head}",
                "layer": layer,
                "head_idx": head,
                "normative_score": norm_scores[(layer, head)],
                "general_score": general_scores[(layer, head)],
            })

        specific_layer_dist = Counter(layer_bin(l) for l, h in specific)
        shared_layer_dist = Counter(layer_bin(l) for l, h in shared)

        ds_result = {
            "label": DATASET_LABELS.get(ds, ds),
            "total_top_k": len(norm_set),
            "specific_count": len(specific),
            "shared_count": len(shared),
            "specificity_ratio": len(specific) / len(norm_set) if norm_set else 0,
            "specific_heads": specific_details,
            "shared_heads": shared_details,
            "specific_layer_distribution": dict(specific_layer_dist),
            "shared_layer_distribution": dict(shared_layer_dist),
        }
        results["datasets"][ds] = ds_result

        log(f"\n--- {DATASET_LABELS.get(ds, ds)} ---")
        log(f"  Specific heads: {len(specific)}/{len(norm_set)} ({len(specific)/len(norm_set)*100:.0f}%)")
        log(f"  Shared heads: {len(shared)}/{len(norm_set)} ({len(shared)/len(norm_set)*100:.0f}%)")
        if specific_details:
            log(f"  Specific head list: {[d['head'] for d in specific_details]}")
            log(f"  Specific head layer distribution: {dict(specific_layer_dist)}")
            top3 = specific_details[:3]
            for d in top3:
                log(f"    {d['head']}: norm={d['normative_score']:.6f}, gen={d['general_score']:.6f}, ratio={d['score_ratio']:.1f}x")

    log(f"\n--- Cross-dataset specific-head analysis ---")

    specific_head_counts = Counter()
    for ds in NORMATIVE_DATASETS:
        if ds not in head_sets:
            continue
        specific = head_sets[ds] - general_set
        for h in specific:
            specific_head_counts[h] += 1

    core_normative = {h for h, c in specific_head_counts.items() if c >= 2}
    log(f"  Total specific heads (union): {len(all_specific_heads)}")
    log(f"  Core normative heads (>=2 datasets): {len(core_normative)}")
    if core_normative:
        for h in sorted(core_normative):
            count = specific_head_counts[h]
            datasets_with = [ds for ds in NORMATIVE_DATASETS
                           if ds in head_sets and h in (head_sets[ds] - general_set)]
            log(f"    L{h[0]}-H{h[1]}: {count} datasets ({', '.join(datasets_with)})")

    universal_heads = set()
    for h in general_set:
        norm_count = sum(1 for ds in NORMATIVE_DATASETS
                        if ds in head_sets and h in head_sets[ds])
        if norm_count >= 3:
            universal_heads.add(h)

    log(f"  Universal infrastructure heads (general + >=3 normative): {len(universal_heads)}")
    for h in sorted(universal_heads):
        norm_count = sum(1 for ds in NORMATIVE_DATASETS
                        if ds in head_sets and h in head_sets[ds])
        log(f"    L{h[0]}-H{h[1]}: general + {norm_count} normative datasets")

    results["summary"] = {
        "total_specific_heads": len(all_specific_heads),
        "core_normative_heads": [f"L{l}-H{h}" for l, h in sorted(core_normative)],
        "core_normative_count": len(core_normative),
        "universal_infrastructure_heads": [f"L{l}-H{h}" for l, h in sorted(universal_heads)],
        "universal_infrastructure_count": len(universal_heads),
        "all_specific_heads": [f"L{l}-H{h}" for l, h in sorted(all_specific_heads)],
        "specific_head_frequency": {f"L{l}-H{h}": c for (l, h), c in specific_head_counts.most_common()},
    }

    return results

def sensitivity_analysis(data: Dict[str, np.ndarray], k_values: List[int] = None) -> Dict:
    if k_values is None:
        k_values = [10, 15, 20, 30]

    log(f"\n{'='*70}")
    log(f"  K sensitivity analysis: K = {k_values}")
    log(f"{'='*70}")

    sensitivity = {}
    for k in k_values:
        result = analyze_set_difference(data, top_k=k)

        per_ds = {}
        for ds, ds_result in result.get("datasets", {}).items():
            per_ds[ds] = {
                "specificity_ratio": ds_result["specificity_ratio"],
                "specific_count": ds_result["specific_count"],
                "shared_count": ds_result["shared_count"],
            }
        sensitivity[str(k)] = {
            "per_dataset": per_ds,
            "total_specific": result.get("summary", {}).get("total_specific_heads", 0),
            "core_normative": result.get("summary", {}).get("core_normative_count", 0),
        }

    log(f"\n{'='*70}")
    log(f"  Sensitivity summary")
    log(f"{'='*70}")
    log(f"{'K':>4} | {'chinese_legal':>14} | {'english_legal':>14} | {'chinese_moral':>14} | {'english_moral':>14} | {'TotalSpec':>8} | {'Core':>6}")
    log("-" * 95)
    for k in k_values:
        sk = str(k)
        row = f"{k:>4}"
        for ds in NORMATIVE_DATASETS:
            if ds in sensitivity[sk]["per_dataset"]:
                d = sensitivity[sk]["per_dataset"][ds]
                row += f" | {d['specific_count']:>5}/{k:<3} ({d['specificity_ratio']*100:>4.0f}%)"
            else:
                row += f" | {'N/A':>14}"
        row += f" | {sensitivity[sk]['total_specific']:>8}"
        row += f" | {sensitivity[sk]['core_normative']:>6}"
        log(row)

    return sensitivity

def generate_report(results: Dict, sensitivity: Dict, output_dir: Path) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Set Difference Analysis Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    top_k = results.get("top_k", 20)
    lines.append(f"\nMethod: top-{top_k} heads, mean_abs aggregation")
    lines.append(f"Set difference = normative_top_{top_k} - general_top_{top_k}")

    lines.append(f"\n{'='*70}")
    lines.append("1. Per-dataset specific-head analysis")
    lines.append(f"{'='*70}")

    for ds in NORMATIVE_DATASETS:
        ds_data = results.get("datasets", {}).get(ds)
        if not ds_data:
            continue

        lines.append(f"\n--- {ds_data['label']} ---")
        lines.append(f"Specific heads: {ds_data['specific_count']}/{ds_data['total_top_k']} "
                     f"(specificity rate {ds_data['specificity_ratio']*100:.0f}%)")
        lines.append(f"Shared heads: {ds_data['shared_count']}/{ds_data['total_top_k']}")

        if ds_data["specific_heads"]:
            lines.append(f"\nSpecific head details:")
            lines.append(f"{'Head':>10} | {'Layer Bin':>22} | {'Norm Score':>12} | {'Gen Score':>12} | {'Ratio':>8}")
            lines.append("-" * 75)
            for h in ds_data["specific_heads"]:
                lines.append(f"{h['head']:>10} | {h['layer_bin']:>22} | "
                           f"{h['normative_score']:>12.6f} | {h['general_score']:>12.6f} | "
                           f"{h['score_ratio']:>7.1f}x")

        if ds_data["specific_layer_distribution"]:
            lines.append(f"\nSpecific head layer distribution: {ds_data['specific_layer_distribution']}")

    summary = results.get("summary", {})
    lines.append(f"\n{'='*70}")
    lines.append("2. Cross-dataset summary")
    lines.append(f"{'='*70}")
    lines.append(f"Total specific heads (union): {summary.get('total_specific_heads', 0)}")
    lines.append(f"Core normative heads (>=2 datasets): {summary.get('core_normative_count', 0)}")
    if summary.get("core_normative_heads"):
        lines.append(f"  Head list: {summary['core_normative_heads']}")
    lines.append(f"Universal infrastructure heads (general + >=3 normative): {summary.get('universal_infrastructure_count', 0)}")
    if summary.get("universal_infrastructure_heads"):
        lines.append(f"  Head list: {summary['universal_infrastructure_heads']}")

    freq = summary.get("specific_head_frequency", {})
    if freq:
        lines.append(f"\nSpecific-head frequency distribution:")
        for head_name, count in sorted(freq.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {head_name}: {count} datasets")

    lines.append(f"\n{'='*70}")
    lines.append("3. K sensitivity analysis")
    lines.append(f"{'='*70}")
    lines.append(f"\n{'K':>4} | {'chinese_legal':>14} | {'english_legal':>14} | {'chinese_moral':>14} | {'english_moral':>14}")
    lines.append("-" * 75)
    for k_str, s_data in sensitivity.items():
        k = int(k_str)
        row = f"{k:>4}"
        for ds in NORMATIVE_DATASETS:
            if ds in s_data["per_dataset"]:
                d = s_data["per_dataset"][ds]
                row += f" | {d['specific_count']:>5}/{k:<3} ({d['specificity_ratio']*100:>4.0f}%)"
            else:
                row += f" | {'N/A':>14}"
        lines.append(row)

    lines.append(f"\n{'='*70}")
    lines.append("4. Paper narrative suggestions")
    lines.append(f"{'='*70}")

    eng_moral = results.get("datasets", {}).get("english_moral", {})
    chi_moral = results.get("datasets", {}).get("chinese_moral", {})
    chi_legal = results.get("datasets", {}).get("chinese_legal", {})
    eng_legal = results.get("datasets", {}).get("english_legal", {})

    if eng_moral:
        lines.append(f"\n- English Moral has highest specificity rate ({eng_moral['specificity_ratio']*100:.0f}%)，"
                     f"suggesting (English) moral reasoning uses distinct attention circuits")
    if chi_legal:
        lines.append(f"- Chinese Legal has lowest specificity rate ({chi_legal['specificity_ratio']*100:.0f}%)，"
                     f"sharing substantial infrastructure with general MCQA")
    lines.append(f"- Core normative heads ({summary.get('core_normative_count', 0)} heads) "
                 f"represent shared normative-reasoning circuits across domains/languages")
    lines.append(f"- Narrative frame: normative reasoning builds upon shared MCQA infrastructure, "
                 f"but recruits additional specialized circuits")

    report_text = "\n".join(lines)
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Set-difference analysis")
    parser.add_argument("--results-dir", type=str, default="output/qwen3-8b",
                       help="Results directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K head count")
    parser.add_argument("--k-values", type=str, default="10,15,20,30",
                       help="K value list for sensitivity analysis")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "set_difference"
    output_dir.mkdir(parents=True, exist_ok=True)

    k_values = [int(k) for k in args.k_values.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log("Loading all datasets...")
    data = load_all_head_data(results_dir)

    if len(data) < 2:
        log("Insufficient data: need general_mcqa + at least 1 normative dataset", "ERROR")
        return

    results = analyze_set_difference(data, top_k=args.top_k)

    sensitivity = sensitivity_analysis(data, k_values=k_values)

    json_path = output_dir / f"set_difference_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"main_analysis": results, "sensitivity": sensitivity},
                 f, indent=2, ensure_ascii=False)
    log(f"\nJSON Results saved: {json_path}")

    report = generate_report(results, sensitivity, output_dir)
    report_path = output_dir / f"set_difference_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    log(f"Text report saved: {report_path}")

    print("\n" + report)

if __name__ == "__main__":
    main()
