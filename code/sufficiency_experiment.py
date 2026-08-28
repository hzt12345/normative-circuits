

import argparse
import json
import os
import sys
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from tqdm import tqdm

import torch
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from head_suppressor import AttentionHeadSuppressor
from reproducibility import set_all_seeds

RANDOM_SEED = 42
AMPLIFICATION_FACTORS = [1.5, 2.0, 3.0]
N_RANDOM_TRIALS = 5

DATASET_CONFIG = {
    "chinese_legal": {
        "results_pattern": "output/qwen3-8b/chinese_legal_atp_results_*.json",
        "name": "Chinese Legal",
    },
    "english_legal": {
        "results_pattern": "output/qwen3-8b/english_legal_atp_results_*.json",
        "name": "English Legal",
    },
    "chinese_moral": {
        "results_pattern": "output/qwen3-8b/chinese_moral_atp_results_*.json",
        "name": "Chinese Moral",
    },
    "english_moral": {
        "results_pattern": "output/qwen3-8b/english_moral_atp_results_*.json",
        "name": "English Moral",
    },
}

def find_latest_file(pattern: str) -> Optional[Path]:
    import glob
    matches = sorted(glob.glob(str(PROJECT_ROOT / pattern)), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None

def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower()

def load_set_difference_results(model_name: str = "Qwen/Qwen3-8B") -> Optional[Dict]:
    sd_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "set_difference"
    if not sd_dir.exists():
        return None
    files = sorted(sd_dir.glob("set_difference_results_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        return None
    with open(files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_head_str(head_str: str) -> Tuple[int, int]:
    parts = head_str.replace("L", "").replace("H", "").split("-")
    return int(parts[0]), int(parts[1])

def get_head_sets(sd_results: Dict) -> Dict[str, List[Tuple[int, int]]]:
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

def get_random_heads(n_heads: int, n_layers: int, n_heads_per_layer: int,
                     exclude: Set[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
    exclude_set = exclude or set()
    all_heads = [
        (layer, head)
        for layer in range(n_layers)
        for head in range(n_heads_per_layer)
        if (layer, head) not in exclude_set
    ]
    return random.sample(all_heads, min(n_heads, len(all_heads)))

def find_incorrect_samples(
    model: HookedTransformer,
    samples: List[Dict],
    device: str = "cuda",
    max_samples: int = 200,
) -> Tuple[List[int], float]:
    print("Finding incorrectly answered samples...")

    incorrect_indices = []
    correct_count = 0
    total = 0

    indices = list(range(min(max_samples, len(samples))))

    for idx in tqdm(indices, desc="Baseline evaluation"):
        sample = samples[idx]["sample"]
        prompt = sample["correct_prompt"]
        expected = sample["answer"]

        tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(tokens)
            probs = torch.softmax(logits[0, -1], dim=-1)

        answer_probs = {}
        for opt in ["A", "B", "C", "D"]:
            opt_ids = model.tokenizer.encode(opt, add_special_tokens=False)
            if opt_ids:
                answer_probs[opt] = probs[opt_ids[0]].item()

        if answer_probs:
            predicted = max(answer_probs, key=answer_probs.get)
            if predicted == expected:
                correct_count += 1
            else:
                incorrect_indices.append(idx)
        total += 1

    baseline_accuracy = correct_count / total if total > 0 else 0.0
    print(f"Baseline accuracy: {baseline_accuracy:.2%} ({correct_count}/{total})")
    print(f"Incorrect samples: {len(incorrect_indices)}")

    return incorrect_indices, baseline_accuracy

def amplify_and_evaluate(
    model: HookedTransformer,
    samples: List[Dict],
    sample_indices: List[int],
    heads_to_amplify: List[Tuple[int, int]],
    amplification_factor: float,
    device: str = "cuda",
) -> Dict:

    suppressor = AttentionHeadSuppressor(
        model=model,
        suppression_strength=amplification_factor,
        suppression_strategy="scale"
    )
    suppressor.set_target_heads(heads_to_amplify)
    suppressor.setup_hooks()

    corrected = 0
    still_wrong = 0
    made_worse = 0
    details = []

    for idx in sample_indices:
        sample = samples[idx]["sample"]
        prompt = sample["correct_prompt"]
        expected = sample["answer"]

        tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = suppressor.run_with_suppression(tokens)
            probs = torch.softmax(logits[0, -1], dim=-1)

        answer_probs = {}
        for opt in ["A", "B", "C", "D"]:
            opt_ids = model.tokenizer.encode(opt, add_special_tokens=False)
            if opt_ids:
                answer_probs[opt] = probs[opt_ids[0]].item()

        if answer_probs:
            predicted = max(answer_probs, key=answer_probs.get)
            if predicted == expected:
                corrected += 1
            else:
                still_wrong += 1
        else:
            still_wrong += 1

        details.append({
            "sample_idx": idx,
            "expected": expected,
            "predicted": predicted if answer_probs else "?",
            "corrected": predicted == expected if answer_probs else False,
        })

    total = len(sample_indices)
    recovery_rate = corrected / total if total > 0 else 0.0

    return {
        "amplification_factor": amplification_factor,
        "n_heads_amplified": len(heads_to_amplify),
        "total_incorrect_samples": total,
        "corrected": corrected,
        "still_wrong": still_wrong,
        "recovery_rate": recovery_rate,
        "details": details,
    }

def run_sufficiency_experiment(
    dataset_key: str,
    max_samples: int = 200,
    output_dir: str = None,
    model_cache: str = "Model",
    device: str = "cuda",
    model_name: str = "Qwen/Qwen3-8B",
) -> Dict:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "sufficiency"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIG[dataset_key]

    print("=" * 70)
    print(f"A5: Sufficiency Experiment - {config['name']}")
    print("=" * 70)

    set_all_seeds(RANDOM_SEED)

    print("\n[1/5] Loading set-difference results...")
    sd_results = load_set_difference_results(model_name=model_name)
    if sd_results is None:
        print("ERROR: set-difference results not found")
        print("Run first: python set_difference_analysis.py")
        return None

    head_sets = get_head_sets(sd_results)
    print(f"  Normative-specific heads: {len(head_sets['normative_specific'])}")
    print(f"  Universal heads: {len(head_sets['universal'])}")

    print("\n[2/5] Loading AtP results...")
    pattern = config["results_pattern"].replace(
        "output/qwen3-8b/", f"output/{model_short_name(model_name)}/", 1
    )
    results_file = find_latest_file(pattern)
    if results_file is None:
        print(f"ERROR: AtP results for {dataset_key} not found (pattern: {pattern})")
        return None

    with open(results_file, 'r', encoding='utf-8') as f:
        atp_data = json.load(f)
    samples = atp_data.get("results", [])
    print(f"  Loaded {len(samples)} samples from {results_file.name}")

    print("\n[3/5] Loading model...")
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, switching to CPU")
        device = "cpu"

    from local_model_loader import load_hooked_transformer
    model = load_hooked_transformer(model_name, device=device)

    n_layers = model.cfg.n_layers
    n_heads_per_layer = model.cfg.n_heads

    print("\n[4/5] Finding incorrect samples...")
    incorrect_indices, baseline_accuracy = find_incorrect_samples(
        model, samples, device, max_samples=max_samples
    )

    if not incorrect_indices:
        print("WARNING: No incorrect samples found (model accuracy = 100%)")
        return None

    if len(incorrect_indices) > 100:
        incorrect_indices = random.sample(incorrect_indices, 100)

    print(f"\n[5/5] Running amplification experiments on {len(incorrect_indices)} incorrect samples...")

    all_results = {
        "metadata": {
            "dataset": dataset_key,
            "dataset_name": config["name"],
            "timestamp": datetime.now().isoformat(),
            "max_samples": max_samples,
            "baseline_accuracy": baseline_accuracy,
            "n_incorrect_samples": len(incorrect_indices),
            "amplification_factors": AMPLIFICATION_FACTORS,
            "n_random_trials": N_RANDOM_TRIALS,
        },
        "head_sets": {
            "normative_specific": [list(h) for h in head_sets["normative_specific"]],
            "universal": [list(h) for h in head_sets["universal"]],
        },
        "experiments": {},
    }

    print("\n--- Amplifying normative-specific heads ---")
    norm_results = {}
    for factor in AMPLIFICATION_FACTORS:
        print(f"  alpha={factor}...")
        result = amplify_and_evaluate(
            model, samples, incorrect_indices,
            head_sets["normative_specific"], factor, device
        )
        norm_results[str(factor)] = result
        print(f"    Recovery: {result['recovery_rate']:.2%} ({result['corrected']}/{result['total_incorrect_samples']})")
    all_results["experiments"]["normative_specific"] = norm_results

    print("\n--- Amplifying universal heads ---")
    univ_results = {}
    if head_sets["universal"]:
        for factor in AMPLIFICATION_FACTORS:
            print(f"  alpha={factor}...")
            result = amplify_and_evaluate(
                model, samples, incorrect_indices,
                head_sets["universal"], factor, device
            )
            univ_results[str(factor)] = result
            print(f"    Recovery: {result['recovery_rate']:.2%} ({result['corrected']}/{result['total_incorrect_samples']})")
    all_results["experiments"]["universal"] = univ_results

    print("\n--- Amplifying random heads ---")
    random_results = {}
    n_norm_heads = len(head_sets["normative_specific"])
    exclude_set = set(head_sets["normative_specific"]) | set(head_sets["universal"])

    for factor in AMPLIFICATION_FACTORS:
        trial_recoveries = []
        for trial in range(N_RANDOM_TRIALS):
            random_heads = get_random_heads(
                n_norm_heads, n_layers, n_heads_per_layer, exclude_set
            )
            result = amplify_and_evaluate(
                model, samples, incorrect_indices,
                random_heads, factor, device
            )
            trial_recoveries.append(result["recovery_rate"])

        random_results[str(factor)] = {
            "mean_recovery": float(np.mean(trial_recoveries)),
            "std_recovery": float(np.std(trial_recoveries)),
            "trials": trial_recoveries,
            "n_heads": n_norm_heads,
        }
        print(f"  alpha={factor}: Recovery = {np.mean(trial_recoveries):.2%} +/- {np.std(trial_recoveries):.2%}")
    all_results["experiments"]["random"] = random_results

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{dataset_key}_sufficiency_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {json_path}")

    report = generate_sufficiency_report(all_results)
    report_path = output_dir / f"{dataset_key}_sufficiency_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    print("\n" + report)

    return all_results

def generate_sufficiency_report(results: Dict) -> str:
    meta = results["metadata"]
    lines = [
        "=" * 70,
        f"A5: Sufficiency Experiment Report - {meta['dataset_name']}",
        "=" * 70,
        f"Generated: {meta['timestamp']}",
        f"Baseline accuracy: {meta['baseline_accuracy']:.2%}",
        f"Incorrect samples tested: {meta['n_incorrect_samples']}",
        "",
        "Recovery rates by head set and amplification factor:",
        "-" * 70,
        f"{'Head Set':<25} {'alpha=1.5':>12} {'alpha=2.0':>12} {'alpha=3.0':>12}",
        "-" * 70,
    ]

    norm = results["experiments"].get("normative_specific", {})
    row = f"{'Normative-specific':<25}"
    for factor in ["1.5", "2.0", "3.0"]:
        if factor in norm:
            row += f" {norm[factor]['recovery_rate']:>11.1%}"
        else:
            row += f" {'N/A':>12}"
    lines.append(row)

    univ = results["experiments"].get("universal", {})
    row = f"{'Universal':<25}"
    for factor in ["1.5", "2.0", "3.0"]:
        if factor in univ:
            row += f" {univ[factor]['recovery_rate']:>11.1%}"
        else:
            row += f" {'N/A':>12}"
    lines.append(row)

    rand = results["experiments"].get("random", {})
    row = f"{'Random (mean +/- std)':<25}"
    for factor in ["1.5", "2.0", "3.0"]:
        if factor in rand:
            m = rand[factor]["mean_recovery"]
            s = rand[factor]["std_recovery"]
            row += f" {m:>5.1%}+/-{s:>4.1%}"
        else:
            row += f" {'N/A':>12}"
    lines.append(row)

    lines.extend([
        "",
        "Interpretation:",
        "-" * 70,
    ])

    for factor in ["2.0"]:
        norm_r = norm.get(factor, {}).get("recovery_rate", 0)
        rand_r = rand.get(factor, {}).get("mean_recovery", 0)
        if norm_r > rand_r:
            lines.append(f"  At alpha={factor}: Normative-specific ({norm_r:.1%}) > Random ({rand_r:.1%})")
            lines.append("  -> Normative-specific heads show sufficiency for normative reasoning")
        else:
            lines.append(f"  At alpha={factor}: Normative-specific ({norm_r:.1%}) <= Random ({rand_r:.1%})")
            lines.append("  -> No clear sufficiency advantage for normative-specific heads")

    lines.append("=" * 70)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(
        description="A5: Sufficiency Experiment (Head Amplification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--dataset", type=str,
                        choices=list(DATASET_CONFIG.keys()),
                        default="chinese_legal")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-cache", type=str, default="Model")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name")

    args = parser.parse_args()

    if args.all_datasets:
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            run_sufficiency_experiment(
                dataset_key=dataset,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                model_cache=args.model_cache,
                device=args.device,
                model_name=args.model,
            )
        except Exception as e:
            print(f"\nERROR processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
