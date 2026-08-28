

import argparse
import json
import os
import sys
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import torch
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from head_suppressor import AttentionHeadSuppressor
from reproducibility import set_all_seeds

KNOCKOUT_K_VALUES = [1, 2, 3, 5, 10, 15, 20]
N_RANDOM_TRIALS = 5
DEFAULT_SUPPRESSION_STRENGTH = 1.0
RANDOM_SEED = 42

DATASET_CONFIG = {
    "chinese_legal": {
        "file_pattern": "data/processed/chinese_legal.json",
        "results_pattern": "output/qwen3-8b/chinese_legal_atp_results_*.json",
        "name": "Chinese Legal",
        "language": "zh",
        "domain": "legal",
    },
    "english_legal": {
        "file_pattern": "data/processed/english_legal.json",
        "results_pattern": "output/qwen3-8b/english_legal_atp_results_*.json",
        "name": "English Legal",
        "language": "en",
        "domain": "legal",
    },
    "chinese_moral": {
        "file_pattern": "data/processed/chinese_moral.json",
        "results_pattern": "output/qwen3-8b/chinese_moral_atp_results_*.json",
        "name": "Chinese Moral",
        "language": "zh",
        "domain": "moral",
    },
    "english_moral": {
        "file_pattern": "data/processed/english_moral.json",
        "results_pattern": "output/qwen3-8b/english_moral_atp_results_*.json",
        "name": "English Moral",
        "language": "en",
        "domain": "moral",
    },
    "general_mcqa": {
        "file_pattern": "data/processed/general_mcqa_stem.json",
        "results_pattern": "output/qwen3-8b/general_mcqa_atp_results_*.json",
        "name": "General MCQA (STEM)",
        "language": "en",
        "domain": "general",
    },
}

def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower()

def load_set_difference_results(model_name: str = "Qwen/Qwen3-8B") -> Optional[Dict]:
    sd_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "set_difference"
    if not sd_dir.exists():
        return None
    files = sorted(sd_dir.glob("set_difference_results_*.json"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_head_str(head_str: str) -> Tuple[int, int]:
    parts = head_str.replace("L", "").replace("H", "").split("-")
    return int(parts[0]), int(parts[1])

def get_selective_head_sets(sd_results: Dict) -> Dict[str, List[Tuple[int, int]]]:
    main = sd_results.get("main_analysis", sd_results)
    summary = main.get("summary", {})

    core_normative = summary.get("core_normative_heads", [])
    normative_specific = [parse_head_str(h) for h in core_normative]

    universal = summary.get("universal_infrastructure_heads", [])
    universal_heads = [parse_head_str(h) for h in universal]

    if len(normative_specific) < 3:
        all_specific = summary.get("all_specific_heads", [])
        normative_specific = [parse_head_str(h) for h in all_specific[:20]]

    return {
        "normative_specific": normative_specific,
        "universal": universal_heads,
    }

def find_latest_file(pattern: str) -> Optional[Path]:
    import glob
    matches = sorted(glob.glob(str(PROJECT_ROOT / pattern)), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None

def load_atp_results(dataset_key: str, model_name: str = "Qwen/Qwen3-8B") -> Dict:
    config = DATASET_CONFIG[dataset_key]

    pattern = config["results_pattern"].replace(
        "output/qwen3-8b/", f"output/{model_short_name(model_name)}/", 1
    )
    results_file = find_latest_file(pattern)

    if not results_file:
        raise FileNotFoundError(f"ATP results for {dataset_key} not found (pattern: {pattern})")

    print(f"Loading ATP results: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_top_heads_from_results(
    results: Dict,
    top_k: int = 20,
    method: str = "mean_abs"
) -> List[Tuple[int, int, float]]:
    samples = results.get("results", [])
    if not samples:
        raise ValueError("No samples in ATP results")

    first_result = samples[0]["patching_result"]
    n_layers = len(first_result)
    first_layer_key = list(first_result.keys())[0]
    n_heads = len(first_result[first_layer_key])
    n_samples = len(samples)

    print(f"Processing {n_samples} samples, {n_layers} layers, {n_heads} heads/layer")

    importance_matrix = np.zeros((n_samples, n_layers, n_heads))

    for sample_idx, sample in enumerate(samples):
        patching_result = sample.get("patching_result")
        if patching_result is None:
            continue
        for layer_idx in range(n_layers):
            layer_key = f"layer_{layer_idx}"
            if layer_key not in patching_result:
                continue
            for head_idx in range(n_heads):
                head_key = f"head_{head_idx}"
                if head_key not in patching_result[layer_key]:
                    continue
                prob_diff = patching_result[layer_key][head_key]["prob_diff"]
                if isinstance(prob_diff, list):
                    prob_diff = np.mean(prob_diff)
                importance_matrix[sample_idx, layer_idx, head_idx] = float(prob_diff)

    head_scores = []
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            values = importance_matrix[:, layer_idx, head_idx]
            if method == "mean_abs":
                score = np.mean(np.abs(values))
            elif method == "max_abs":
                score = np.max(np.abs(values))
            elif method == "mean":
                score = np.mean(values)
            else:
                score = np.mean(np.abs(values))
            head_scores.append((layer_idx, head_idx, score))

    head_scores.sort(key=lambda x: x[2], reverse=True)

    return head_scores[:top_k]

def get_random_heads(
    n_heads: int,
    n_layers: int,
    n_heads_per_layer: int,
    exclude: List[Tuple[int, int]] = None
) -> List[Tuple[int, int]]:
    exclude_set = set(exclude) if exclude else set()
    all_heads = [
        (layer, head)
        for layer in range(n_layers)
        for head in range(n_heads_per_layer)
        if (layer, head) not in exclude_set
    ]
    return random.sample(all_heads, min(n_heads, len(all_heads)))

class KnockoutEvaluator:

    def __init__(
        self,
        model: HookedTransformer,
        samples: List[Dict],
        device: str = "cuda"
    ):
        self.model = model
        self.samples = samples
        self.device = device
        self.tokenizer = model.tokenizer

    def evaluate_accuracy(
        self,
        suppressor: Optional[AttentionHeadSuppressor] = None,
        sample_indices: Optional[List[int]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        if sample_indices is None:
            sample_indices = list(range(len(self.samples)))

        correct = 0
        total = 0
        results = []

        iterator = tqdm(sample_indices, desc="Evaluating") if show_progress else sample_indices

        for idx in iterator:
            sample = self.samples[idx]["sample"]
            correct_prompt = sample["correct_prompt"]
            expected_answer = sample["answer"]

            tokens = self.tokenizer.encode(correct_prompt, return_tensors="pt").to(self.device)

            answer_token_ids = self.tokenizer.encode(expected_answer, add_special_tokens=False)
            if not answer_token_ids:
                continue
            answer_token_id = answer_token_ids[0]

            with torch.no_grad():
                if suppressor is not None and suppressor.target_heads:
                    suppressor.setup_hooks()
                    logits = suppressor.run_with_suppression(tokens)
                else:
                    logits = self.model(tokens)

                probs = torch.softmax(logits[0, -1], dim=-1)

                answer_options = ["A", "B", "C", "D"]
                answer_probs = {}
                for opt in answer_options:
                    opt_ids = self.tokenizer.encode(opt, add_special_tokens=False)
                    if opt_ids:
                        answer_probs[opt] = probs[opt_ids[0]].item()

                if answer_probs:
                    predicted_answer = max(answer_probs, key=answer_probs.get)
                    is_correct = predicted_answer == expected_answer
                else:
                    is_correct = False
                    predicted_answer = "?"

            correct += int(is_correct)
            total += 1

            results.append({
                "sample_idx": idx,
                "expected": expected_answer,
                "predicted": predicted_answer,
                "correct": is_correct,
                "answer_probs": answer_probs
            })

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results
        }

def run_effect_rank_knockout(
    evaluator: KnockoutEvaluator,
    model: HookedTransformer,
    ranked_heads: List[Tuple[int, int, float]],
    k_values: List[int],
    sample_indices: List[int],
    suppression_strength: float = 1.0
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Effect-rank knockout experiment")
    print("=" * 60)

    results = {
        "type": "effect_rank",
        "k_values": k_values,
        "results_by_k": {}
    }

    for k in k_values:
        if k > len(ranked_heads):
            print(f"Skipping k={k} (exceeds available head count {len(ranked_heads)})")
            continue

        top_k_heads = [(h[0], h[1]) for h in ranked_heads[:k]]

        print(f"\nKnockout Top-{k} heads:")
        for i, (layer, head, score) in enumerate(ranked_heads[:k]):
            print(f"  {i+1}. L{layer}H{head} (score={score:.4f})")

        suppressor = AttentionHeadSuppressor(
            model=model,
            suppression_strength=suppression_strength,
            suppression_strategy="multiply"
        )
        suppressor.set_target_heads(top_k_heads)

        eval_result = evaluator.evaluate_accuracy(
            suppressor=suppressor,
            sample_indices=sample_indices,
            show_progress=True
        )

        results["results_by_k"][k] = {
            "heads": top_k_heads,
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"]
        }

        print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

    return results

def run_random_rank_knockout(
    evaluator: KnockoutEvaluator,
    model: HookedTransformer,
    k_values: List[int],
    sample_indices: List[int],
    n_layers: int,
    n_heads_per_layer: int,
    n_trials: int = 5,
    suppression_strength: float = 1.0,
    exclude_heads: List[Tuple[int, int]] = None
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print(f"Random-rank knockout experiment ({n_trials} trials)")
    print("=" * 60)

    results = {
        "type": "random_rank",
        "k_values": k_values,
        "n_trials": n_trials,
        "results_by_k": {}
    }

    for k in k_values:
        print(f"\nRandom knockout {k} heads...")

        trial_accuracies = []
        trial_details = []

        for trial in range(n_trials):

            random_heads = get_random_heads(
                n_heads=k,
                n_layers=n_layers,
                n_heads_per_layer=n_heads_per_layer,
                exclude=exclude_heads
            )

            suppressor = AttentionHeadSuppressor(
                model=model,
                suppression_strength=suppression_strength,
                suppression_strategy="multiply"
            )
            suppressor.set_target_heads(random_heads)

            eval_result = evaluator.evaluate_accuracy(
                suppressor=suppressor,
                sample_indices=sample_indices,
                show_progress=False
            )

            trial_accuracies.append(eval_result["accuracy"])
            trial_details.append({
                "trial": trial + 1,
                "heads": random_heads,
                "accuracy": eval_result["accuracy"],
                "correct": eval_result["correct"],
                "total": eval_result["total"]
            })

            print(f"  Trial {trial+1}: {eval_result['accuracy']:.2%}")

        results["results_by_k"][k] = {
            "mean_accuracy": np.mean(trial_accuracies),
            "std_accuracy": np.std(trial_accuracies),
            "min_accuracy": np.min(trial_accuracies),
            "max_accuracy": np.max(trial_accuracies),
            "trials": trial_details
        }

        print(f"  Mean accuracy: {np.mean(trial_accuracies):.2%} ± {np.std(trial_accuracies):.2%}")

    return results

def run_baseline_evaluation(
    evaluator: KnockoutEvaluator,
    sample_indices: List[int]
) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("Baseline evaluation (no suppression)")
    print("=" * 60)

    eval_result = evaluator.evaluate_accuracy(
        suppressor=None,
        sample_indices=sample_indices,
        show_progress=True
    )

    print(f"Baseline accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

    return {
        "accuracy": eval_result["accuracy"],
        "correct": eval_result["correct"],
        "total": eval_result["total"]
    }

def generate_knockout_plot(
    baseline_accuracy: float,
    effect_rank_results: Dict,
    random_rank_results: Dict,
    output_path: str,
    dataset_name: str = "Dataset"
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Warning: plotly not installed; skipping chart")
        return

    k_values = effect_rank_results["k_values"]

    effect_accuracies = []
    random_means = []
    random_stds = []

    for k in k_values:
        if k in effect_rank_results["results_by_k"]:
            effect_accuracies.append(effect_rank_results["results_by_k"][k]["accuracy"])
        else:
            effect_accuracies.append(None)

        if k in random_rank_results["results_by_k"]:
            random_means.append(random_rank_results["results_by_k"][k]["mean_accuracy"])
            random_stds.append(random_rank_results["results_by_k"][k]["std_accuracy"])
        else:
            random_means.append(None)
            random_stds.append(None)

    valid_k = [k for i, k in enumerate(k_values) if effect_accuracies[i] is not None]
    valid_effect = [a for a in effect_accuracies if a is not None]
    valid_random_mean = [m for m in random_means if m is not None]
    valid_random_std = [s for s in random_stds if s is not None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0] + valid_k,
        y=[baseline_accuracy] * (len(valid_k) + 1),
        mode='lines',
        name='Baseline',
        line=dict(color='green', dash='dash', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[0] + valid_k,
        y=[baseline_accuracy] + valid_effect,
        mode='lines+markers',
        name='Effect-Rank Knockout',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))

    if valid_random_mean:
        fig.add_trace(go.Scatter(
            x=[0] + valid_k,
            y=[baseline_accuracy] + valid_random_mean,
            mode='lines+markers',
            name='Random-Rank Knockout',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=[0] + valid_random_std,
                visible=True
            )
        ))

    fig.update_layout(
        title=dict(
            text=f'Knockout Validation - {dataset_name}',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='Number of Knocked-out Heads (K)',
            tickmode='array',
            tickvals=[0] + valid_k,
            ticktext=['0'] + [str(k) for k in valid_k]
        ),
        yaxis=dict(
            title='Accuracy',
            tickformat='.0%',
            range=[0, 1.05]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        template='plotly_white',
        width=800,
        height=500
    )

    fig.write_html(output_path)
    print(f"Chart saved: {output_path}")

    try:
        png_path = output_path.replace('.html', '.png')
        fig.write_image(png_path)
        print(f"PNG saved: {png_path}")
    except Exception as e:
        print(f"Warning: cannot save PNG (needs kaleido): {e}")

def generate_summary_report(
    dataset_name: str,
    baseline: Dict,
    effect_rank: Dict,
    random_rank: Dict,
    output_path: str
):
    lines = [
        "=" * 70,
        f"Knockout Validation Report - {dataset_name}",
        "=" * 70,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "1. Baseline",
        "-" * 40,
        f"   Accuracy: {baseline['accuracy']:.2%} ({baseline['correct']}/{baseline['total']})",
        "",
        "2. Effect-rank knockout (importance order)",
        "-" * 40,
    ]

    for k, result in sorted(effect_rank["results_by_k"].items()):
        acc = result["accuracy"]
        drop = baseline["accuracy"] - acc
        drop_pct = (drop / baseline["accuracy"]) * 100 if baseline["accuracy"] > 0 else 0
        lines.append(f"   K={k:2d}: {acc:.2%} (drop {drop_pct:.1f}%)")

    lines.extend([
        "",
        f"3. Random-rank knockout (random order, {random_rank['n_trials']} trials)",
        "-" * 40,
    ])

    for k, result in sorted(random_rank["results_by_k"].items()):
        mean_acc = result["mean_accuracy"]
        std_acc = result["std_accuracy"]
        drop = baseline["accuracy"] - mean_acc
        drop_pct = (drop / baseline["accuracy"]) * 100 if baseline["accuracy"] > 0 else 0
        lines.append(f"   K={k:2d}: {mean_acc:.2%} ± {std_acc:.2%} (drop {drop_pct:.1f}%)")

    lines.extend([
        "",
        "4. Validation analysis",
        "-" * 40,
    ])

    validation_passed = True
    for k in effect_rank["results_by_k"].keys():
        if k in random_rank["results_by_k"]:
            effect_acc = effect_rank["results_by_k"][k]["accuracy"]
            random_acc = random_rank["results_by_k"][k]["mean_accuracy"]

            effect_drop = baseline["accuracy"] - effect_acc
            random_drop = baseline["accuracy"] - random_acc

            is_significant = effect_drop > random_drop * 1.5

            status = "✓" if is_significant else "✗"
            lines.append(f"   K={k}: effect-rank drop {effect_drop:.2%} vs random-rank drop {random_drop:.2%} {status}")

            if not is_significant and k >= 5:
                validation_passed = False

    lines.extend([
        "",
        "5. Conclusion",
        "-" * 40,
        f"   Validation: {'PASS' if validation_passed else 'NEEDS REVIEW'}",
        "",
        "   Interpretation:",
        "   - if effect-rank drop >> random-rank drop, the identified heads are causally important",
        "   - Expected: effect-rank should drop >30% at K>=5",
        "   - Expected: random-rank drop <10%",
        "",
        "=" * 70,
    ])

    report = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved: {output_path}")
    print("\n" + report)

    return report

def run_selective_knockout(
    evaluator: KnockoutEvaluator,
    model: HookedTransformer,
    head_set: List[Tuple[int, int]],
    head_set_name: str,
    sample_indices: List[int],
    suppression_strength: float = 1.0
) -> Dict[str, Any]:
    print(f"\n--- Selective Knockout: {head_set_name} ({len(head_set)} heads) ---")

    if not head_set:
        print(f"  WARNING: {head_set_name} head set is empty, skipping")
        return {"accuracy": None, "heads": [], "n_heads": 0}

    for i, (layer, head) in enumerate(head_set[:10]):
        print(f"  {i+1}. L{layer}H{head}")
    if len(head_set) > 10:
        print(f"  ... ({len(head_set) - 10} more)")

    suppressor = AttentionHeadSuppressor(
        model=model,
        suppression_strength=suppression_strength,
        suppression_strategy="multiply"
    )
    suppressor.set_target_heads(head_set)

    eval_result = evaluator.evaluate_accuracy(
        suppressor=suppressor,
        sample_indices=sample_indices,
        show_progress=True
    )

    print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

    return {
        "head_set_name": head_set_name,
        "n_heads": len(head_set),
        "heads": [list(h) for h in head_set],
        "accuracy": eval_result["accuracy"],
        "correct": eval_result["correct"],
        "total": eval_result["total"],
    }

def run_knockout_validation(
    dataset_key: str,
    max_samples: int = 100,
    k_values: List[int] = None,
    n_random_trials: int = 5,
    output_dir: str = None,
    model_cache: str = "Model",
    device: str = "cuda",
    head_set: str = None,
    model_name: str = "Qwen/Qwen3-8B",
):
    if k_values is None:
        k_values = KNOCKOUT_K_VALUES

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "knockout"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Knockout Validation Experiment - {DATASET_CONFIG[dataset_key]['name']}")
    if head_set:
        print(f"Mode: A2 Selective Knockout (head_set={head_set})")
    print("=" * 70)
    print(f"Max samples: {max_samples}")
    print(f"K values: {k_values}")
    print(f"Random trials: {n_random_trials}")
    print(f"Output directory: {output_dir}")

    set_all_seeds(RANDOM_SEED)

    print("\n[1/5] Loading ATP analysis results...")
    atp_results = load_atp_results(dataset_key, model_name=model_name)

    print("\n[2/5] Extracting important head ranking...")
    max_k = max(k_values)
    ranked_heads = extract_top_heads_from_results(atp_results, top_k=max_k + 10)

    print(f"\nTop-{min(10, len(ranked_heads))} important heads:")
    for i, (layer, head, score) in enumerate(ranked_heads[:10]):
        print(f"  {i+1}. Layer {layer}, Head {head}: {score:.6f}")

    print("\n[3/5] Loading model...")
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; switching to CPU")
        device = "cpu"

    from local_model_loader import load_hooked_transformer
    model = load_hooked_transformer(model_name, device=device)

    n_layers = model.cfg.n_layers
    n_heads_per_layer = model.cfg.n_heads
    print(f"Model: {n_layers} layers, {n_heads_per_layer} heads/layer")

    print("\n[4/5] Preparing evaluation data...")
    samples = atp_results.get("results", [])

    if max_samples and max_samples < len(samples):
        sample_indices = random.sample(range(len(samples)), max_samples)
    else:
        sample_indices = list(range(len(samples)))

    print(f"Evaluation samples: {len(sample_indices)}")

    evaluator = KnockoutEvaluator(
        model=model,
        samples=samples,
        device=device
    )

    print("\n[5/5] Running knockout experiment...")

    baseline_result = run_baseline_evaluation(evaluator, sample_indices)

    if head_set:
        sd_results = load_set_difference_results(model_name=model_name)
        if sd_results is None:
            print("ERROR: set-difference results not found; run set_difference_analysis.py first")
            return None

        head_sets = get_selective_head_sets(sd_results)
        selective_results = {}

        sets_to_test = []
        if head_set in ("normative_specific", "all"):
            sets_to_test.append(("normative_specific", head_sets["normative_specific"]))
        if head_set in ("universal", "all"):
            sets_to_test.append(("universal", head_sets["universal"]))
        if head_set in ("random", "all"):

            n_norm = len(head_sets["normative_specific"])
            exclude = set(head_sets["normative_specific"]) | set(head_sets["universal"])
            for trial in range(n_random_trials):
                random_heads = get_random_heads(
                    n_norm, n_layers, n_heads_per_layer, list(exclude)
                )
                sets_to_test.append((f"random_trial_{trial+1}", random_heads))

        for set_name, heads in sets_to_test:
            result = run_selective_knockout(
                evaluator, model, heads, set_name,
                sample_indices, DEFAULT_SUPPRESSION_STRENGTH
            )
            selective_results[set_name] = result

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_results = {
            "metadata": {
                "dataset": dataset_key,
                "dataset_name": DATASET_CONFIG[dataset_key]["name"],
                "timestamp": datetime.now().isoformat(),
                "mode": "selective_knockout",
                "head_set": head_set,
                "max_samples": max_samples,
                "actual_samples": len(sample_indices),
            },
            "baseline": baseline_result,
            "selective_knockout": selective_results,
        }

        json_path = output_dir / f"{dataset_key}_selective_knockout_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved: {json_path}")

        report_lines = [
            "=" * 70,
            f"A2: Selective Knockout Report - {DATASET_CONFIG[dataset_key]['name']}",
            "=" * 70,
            f"Baseline accuracy: {baseline_result['accuracy']:.2%}",
            "",
            f"{'Head Set':<30} {'Accuracy':>10} {'Drop':>10} {'N Heads':>10}",
            "-" * 65,
        ]

        for set_name, result in selective_results.items():
            if result["accuracy"] is not None:
                drop = baseline_result["accuracy"] - result["accuracy"]
                drop_pct = (drop / baseline_result["accuracy"] * 100) if baseline_result["accuracy"] > 0 else 0
                report_lines.append(
                    f"{set_name:<30} {result['accuracy']:>9.2%} {drop_pct:>9.1f}% {result['n_heads']:>10}"
                )

        random_results = [v for k, v in selective_results.items() if k.startswith("random_")]
        if random_results:
            accs = [r["accuracy"] for r in random_results if r["accuracy"] is not None]
            if accs:
                mean_acc = np.mean(accs)
                drop = baseline_result["accuracy"] - mean_acc
                drop_pct = (drop / baseline_result["accuracy"] * 100) if baseline_result["accuracy"] > 0 else 0
                report_lines.append(f"{'random (mean)':<30} {mean_acc:>9.2%} {drop_pct:>9.1f}%")

        report_lines.append("=" * 70)
        report = "\n".join(report_lines)

        report_path = output_dir / f"{dataset_key}_selective_knockout_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print("\n" + report)

        return full_results

    effect_rank_result = run_effect_rank_knockout(
        evaluator=evaluator,
        model=model,
        ranked_heads=ranked_heads,
        k_values=k_values,
        sample_indices=sample_indices,
        suppression_strength=DEFAULT_SUPPRESSION_STRENGTH
    )

    random_rank_result = run_random_rank_knockout(
        evaluator=evaluator,
        model=model,
        k_values=k_values,
        sample_indices=sample_indices,
        n_layers=n_layers,
        n_heads_per_layer=n_heads_per_layer,
        n_trials=n_random_trials,
        suppression_strength=DEFAULT_SUPPRESSION_STRENGTH,
        exclude_heads=[(h[0], h[1]) for h in ranked_heads[:max_k]]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    full_results = {
        "metadata": {
            "dataset": dataset_key,
            "dataset_name": DATASET_CONFIG[dataset_key]["name"],
            "timestamp": datetime.now().isoformat(),
            "max_samples": max_samples,
            "actual_samples": len(sample_indices),
            "k_values": k_values,
            "n_random_trials": n_random_trials,
            "suppression_strength": DEFAULT_SUPPRESSION_STRENGTH,
            "random_seed": RANDOM_SEED
        },
        "ranked_heads": [(h[0], h[1], h[2]) for h in ranked_heads],
        "baseline": baseline_result,
        "effect_rank": effect_rank_result,
        "random_rank": random_rank_result
    }

    json_path = output_dir / f"{dataset_key}_knockout_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {json_path}")

    plot_path = output_dir / f"{dataset_key}_knockout_comparison_{timestamp}.html"
    generate_knockout_plot(
        baseline_accuracy=baseline_result["accuracy"],
        effect_rank_results=effect_rank_result,
        random_rank_results=random_rank_result,
        output_path=str(plot_path),
        dataset_name=DATASET_CONFIG[dataset_key]["name"]
    )

    report_path = output_dir / f"{dataset_key}_knockout_report_{timestamp}.txt"
    generate_summary_report(
        dataset_name=DATASET_CONFIG[dataset_key]["name"],
        baseline=baseline_result,
        effect_rank=effect_rank_result,
        random_rank=random_rank_result,
        output_path=str(report_path)
    )

    return full_results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knockout Validation Experiment - Zhang et al. 2024 Figure 3 Style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        default="chinese_legal",
        help="Dataset to test"
    )

    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all datasets"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max evaluation samples (default: 100)"
    )

    parser.add_argument(
        "--k-values",
        type=str,
        default="1,2,3,5,10,15,20",
        help="K values to test, comma-separated (default: 1,2,3,5,10,15,20)"
    )

    parser.add_argument(
        "--n-random-trials",
        type=int,
        default=5,
        help="Random control trials (default: 5)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/qwen3-8b/knockout/)"
    )

    parser.add_argument(
        "--model-cache",
        type=str,
        default="Model",
        help="Model cache directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device"
    )

    parser.add_argument(
        "--head-set",
        type=str,
        choices=["normative_specific", "universal", "random", "all"],
        default=None,
        help="A2: Head set for selective knockout (run set_difference_analysis.py first)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.1)"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    if args.all_datasets:
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            run_knockout_validation(
                dataset_key=dataset,
                max_samples=args.max_samples,
                k_values=k_values,
                n_random_trials=args.n_random_trials,
                output_dir=args.output_dir,
                model_cache=args.model_cache,
                device=args.device,
                head_set=args.head_set,
                model_name=args.model,
            )
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
