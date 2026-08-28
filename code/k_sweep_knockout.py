

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from local_model_loader import load_hooked_transformer
from reproducibility import set_all_seeds
from resampling_knockout import (
    DATASET_CONFIG, RANDOM_SEED, N_RANDOM_TRIALS,
    model_short_name, find_latest_file,
    load_samples, get_answer_token_ids, per_sample_metrics,
    cache_z_for_heads, make_mean_hook, precompute_mean_z,
    get_random_heads_excluding, _aggregate,
)

K_VALUES_DEFAULT = [1, 2, 3, 5, 10]

def load_per_dataset_ns_heads_ranked(
    dataset_key: str,
    model_name: str,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    sd_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "set_difference"
    files = sorted(sd_dir.glob("set_difference_results_*.json"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No set_difference_results_*.json in {sd_dir}")
    with open(files[0], 'r', encoding='utf-8') as f:
        sd = json.load(f)
    datasets = sd["main_analysis"]["datasets"]

    if dataset_key == "general_mcqa":
        per_dataset = []
        for name, ds in datasets.items():
            for h in ds.get("specific_heads", []):
                per_dataset.append((h["layer"], h["head_idx"], float(h.get("normative_score", 0.0))))
        per_dataset = list({(l, h): (l, h, s) for l, h, s in per_dataset}.values())
    else:
        ds_data = datasets.get(dataset_key)
        if ds_data is None:
            raise KeyError(f"Dataset {dataset_key} not in set_difference results")
        per_dataset = [
            (h["layer"], h["head_idx"], float(h.get("score_ratio", h.get("normative_score", 0.0))))
            for h in ds_data["specific_heads"]
        ]

    per_dataset.sort(key=lambda x: -x[2])
    all_ns_union = set()
    for name, ds in datasets.items():
        for h in ds.get("specific_heads", []):
            all_ns_union.add((h["layer"], h["head_idx"]))
    return per_dataset, sorted(all_ns_union)

def evaluate_with_mean_ablation(
    model, tokenizer, samples, target_heads, mean_z_cache, device, desc,
):
    option_ids = get_answer_token_ids(tokenizer)
    heads_by_layer = {}
    for l, h in target_heads:
        heads_by_layer.setdefault(l, []).append(h)

    per_sample = []
    for sample in tqdm(samples, desc=desc):
        c_tokens = tokenizer.encode(sample["correct_prompt"], return_tensors="pt").to(device)
        hooks = []
        for l, h_list in heads_by_layer.items():
            hooks.append((
                f"blocks.{l}.attn.hook_z",
                make_mean_hook(l, h_list, mean_z_cache[l]),
            ))
        with torch.no_grad():
            logits = model.run_with_hooks(c_tokens, fwd_hooks=hooks)
        m = per_sample_metrics(logits, option_ids, sample["answer"])
        per_sample.append(m)
        del c_tokens, logits
        torch.cuda.empty_cache() if device == "cuda" else None
    return _aggregate(per_sample)

def run_k_sweep_cell(
    model, tokenizer,
    dataset_key: str, model_name: str,
    max_samples: int = 500,
    k_values: List[int] = None,
    n_random_trials: int = 5,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    mean_pool_n: int = 100,
):
    if k_values is None:
        k_values = K_VALUES_DEFAULT

    set_all_seeds(RANDOM_SEED)
    dataset_seed = RANDOM_SEED + abs(hash(dataset_key)) % 100000

    ranked_ns, ns_union = load_per_dataset_ns_heads_ranked(dataset_key, model_name)
    full_k = len(ranked_ns)
    print(f"[{dataset_key}] Per-dataset NS (importance-ranked) K_full={full_k}")
    for l, h, s in ranked_ns:
        print(f"  L{l}-H{h}  score={s:.3f}")

    k_values = [k for k in k_values if k <= full_k]
    if full_k not in k_values:
        k_values.append(full_k)
    k_values = sorted(set(k_values))
    print(f"[{dataset_key}] K values: {k_values}")

    samples = load_samples(dataset_key, max_samples=max_samples)
    print(f"[{dataset_key}] N={len(samples)} samples")

    n_layers = model.cfg.n_layers
    n_heads_per_layer = model.cfg.n_heads

    option_ids = get_answer_token_ids(tokenizer)
    per_sample_base = []
    for s in tqdm(samples, desc=f"baseline {dataset_key}"):
        t = tokenizer.encode(s["correct_prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(t)
        per_sample_base.append(per_sample_metrics(logits, option_ids, s["answer"]))
    baseline = _aggregate(per_sample_base)
    print(f"[baseline] acc={baseline['accuracy']:.4f} logit_diff={baseline['mean_logit_diff']:+.4f}")

    effect_layers = sorted(set(l for l, _, _ in ranked_ns))
    print(f"[mean-precompute] effect-rank layers: {effect_layers}")
    mean_z_effect = precompute_mean_z(
        model, tokenizer, samples[:mean_pool_n], effect_layers,
        device=device, n_pool=mean_pool_n,
    )

    results_by_k = {}
    for k in k_values:

        effect_heads = [(l, h) for l, h, _ in ranked_ns[:k]]
        eff_result = evaluate_with_mean_ablation(
            model, tokenizer, samples, effect_heads, mean_z_effect,
            device=device, desc=f"effect-K={k} {dataset_key}",
        )
        print(f"[K={k} effect] acc={eff_result['accuracy']:.4f} logit_diff={eff_result['mean_logit_diff']:+.4f}")

        rng = random.Random(dataset_seed + k)
        rand_trials = []
        for trial in range(n_random_trials):
            rh = get_random_heads_excluding(k, n_layers, n_heads_per_layer, ns_union, rng)
            r_layers = sorted(set(l for l, _ in rh))
            mean_z_r = precompute_mean_z(
                model, tokenizer, samples[:mean_pool_n], r_layers,
                device=device, n_pool=mean_pool_n,
            )
            r_result = evaluate_with_mean_ablation(
                model, tokenizer, samples, rh, mean_z_r,
                device=device, desc=f"rand-K={k} t={trial+1} {dataset_key}",
            )
            print(f"[K={k} random {trial+1}] acc={r_result['accuracy']:.4f} logit_diff={r_result['mean_logit_diff']:+.4f}")
            rand_trials.append(r_result)
            del mean_z_r

        results_by_k[str(k)] = {
            "effect": eff_result,
            "effect_heads": [list(h) for h in effect_heads],
            "random_trials": rand_trials,
        }

    out = {
        "metadata": {
            "model": model_name, "dataset": dataset_key,
            "max_samples": max_samples, "actual_samples": len(samples),
            "n_random_trials": n_random_trials,
            "ablation": "mean", "k_values": k_values,
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED, "dataset_seed": dataset_seed,
        },
        "ranked_ns": [[l, h, s] for l, h, s in ranked_ns],
        "ns_union_excluded_from_random": [list(h) for h in ns_union],
        "baseline": baseline,
        "results_by_k": results_by_k,
    }

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "knockout"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"{dataset_key}_iter05_ksweep_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="chinese_legal")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--n-random-trials", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mean-pool-n", type=int, default=100)
    p.add_argument("--k-values", default="1,2,3,5,10",
                   help="comma-separated K values (full_NS_size added automatically)")
    args = p.parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]

    datasets = list(DATASET_CONFIG.keys()) if args.all_datasets else [args.dataset]
    model = load_hooked_transformer(args.model, device=args.device)
    print(f"[main] Model loaded; running K-sweep on {len(datasets)} dataset(s)")

    for ds in datasets:
        try:
            run_k_sweep_cell(
                model=model, tokenizer=model.tokenizer,
                dataset_key=ds, model_name=args.model,
                max_samples=args.max_samples,
                k_values=k_values,
                n_random_trials=args.n_random_trials,
                device=args.device,
                mean_pool_n=args.mean_pool_n,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"ERROR on {ds}: {e}")

if __name__ == "__main__":
    main()
