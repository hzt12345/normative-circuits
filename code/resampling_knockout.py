#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resampling Knockout with Per-Dataset NS Heads.

Methodological motivation (pre-registered in chat 2026-05-18):
  iter-02 N=500 zero-ablation knockout on pooled NS (≥2 datasets) returned null
  in 12/15 cells. Two suspected confounds:

  (A) Pooled NS (heads important in ≥2 normative datasets) is the wrong instrument
      for testing necessity on any single dataset — it discards heads that are
      specifically critical for one dataset. Per-dataset NS (heads in this
      dataset's top-20 minus general MCQA top-20) is the right set.

  (B) Zero ablation injects an off-distribution residual contribution. Random
      controls show ~-2pp accuracy drops just from off-distribution noise. This
      depresses the signal-to-noise of NS-vs-random gap. Resampling ablation
      (replace head output with its activation on the same sample's
      counterfactual prompt) keeps activations on-distribution.

Hypothesis: under (A+B), NS-specific necessity emerges in ≥8/15 cells with
Bonferroni-corrected significance (α=0.05, N=15).

Method per cell (model × dataset, N=500 samples, K = per-dataset NS set size):
  1. Run baseline on correct_prompt → baseline accuracy
  2. For each sample, run incorrect_prompt and cache all hook_z activations
  3. Run NS knockout: replace per-dataset NS heads' z with cached counterfactual
     z on correct_prompt forward, measure accuracy
  4. Run random control × 5 trials: K random heads (excluding any NS heads in
     this model across all datasets), same resampling procedure
  5. Two-proportion z-test NS vs random-pooled, Bonferroni across 15 cells

Usage:
  python resampling_knockout.py --dataset chinese_legal --model Qwen/Qwen3-8B
  python resampling_knockout.py --all-datasets --model Qwen/Qwen3-8B
"""

import argparse
import hashlib
import json
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
from batch_runner import run_dataset_batch
from head_sampling import (
    ensure_nonempty_head_set,
    get_layer_matched_random_heads_excluding,
    get_random_heads_excluding,
)

RANDOM_SEED = 42
N_RANDOM_TRIALS = 5

DATASET_CONFIG = {
    "chinese_legal": {
        "file_pattern": "data/processed/chinese_legal.json",
        "name": "中文法律",
    },
    "english_legal": {
        "file_pattern": "data/processed/english_legal.json",
        "name": "英文法律",
    },
    "chinese_moral": {
        "file_pattern": "data/processed/chinese_moral.json",
        "name": "中文道德",
    },
    "english_moral": {
        "file_pattern": "data/processed/english_moral.json",
        "name": "英文道德",
    },
    "general_mcqa": {
        "file_pattern": "data/processed/general_mcqa_stem.json",
        "name": "通用MCQA(STEM)",
    },
}


def model_short_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower()


def find_latest_file(pattern: str) -> Optional[Path]:
    import glob
    matches = sorted(glob.glob(str(PROJECT_ROOT / pattern)), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def parse_head_str(head_str: str) -> Tuple[int, int]:
    parts = head_str.replace("L", "").replace("H", "").split("-")
    return int(parts[0]), int(parts[1])


def load_per_dataset_ns_heads(
    dataset_key: str,
    model_name: str,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Return (per-dataset NS heads, all-NS-union for exclusion).

    Per-dataset NS = top-20 of this dataset minus top-20 general MCQA.
    All-NS-union = union of all per-dataset NS across the 4 normative datasets
                   (used to exclude these from the random control sampling).
    """
    sd_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "set_difference"
    files = sorted(sd_dir.glob("set_difference_results_*.json"),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No set_difference_results_*.json in {sd_dir}")

    with open(files[0], 'r', encoding='utf-8') as f:
        sd = json.load(f)

    datasets = sd["main_analysis"]["datasets"]

    if dataset_key == "general_mcqa":
        # For general MCQA cell, "NS" is the per-dataset NS of the dataset we are
        # nominally testing — but there is no NS for general MCQA itself.
        # Use the union NS as the head set (this measures whether normative-
        # specific heads damage general MCQA, our collateral baseline).
        per_dataset_ns = []
        for name, ds in datasets.items():
            for h in ds.get("specific_heads", []):
                per_dataset_ns.append((h["layer"], h["head_idx"]))
        per_dataset_ns = list(set(per_dataset_ns))
    else:
        ds_data = datasets.get(dataset_key)
        if ds_data is None:
            raise KeyError(f"Dataset {dataset_key} not in set_difference results")
        per_dataset_ns = [(h["layer"], h["head_idx"]) for h in ds_data["specific_heads"]]

    all_ns_union = set()
    for name, ds in datasets.items():
        for h in ds.get("specific_heads", []):
            all_ns_union.add((h["layer"], h["head_idx"]))

    return per_dataset_ns, sorted(all_ns_union)


def load_samples(dataset_key: str, max_samples: int = 500) -> List[Dict]:
    pattern = DATASET_CONFIG[dataset_key]["file_pattern"]
    f = find_latest_file(pattern)
    if not f:
        raise FileNotFoundError(f"No data file matching {pattern}")
    print(f"[data] Loading {f}")
    with open(f, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("data", data.get("samples", []))
    set_all_seeds(RANDOM_SEED)
    if max_samples and max_samples < len(data):
        data = random.sample(data, max_samples)
    return data


def get_answer_token_ids(tokenizer, options=("A", "B", "C", "D")) -> Dict[str, int]:
    out = {}
    for opt in options:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        if ids:
            out[opt] = ids[0]
    return out


def predict_answer(logits: torch.Tensor, option_ids: Dict[str, int]) -> str:
    """argmax over the option token ids at the last position."""
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    scores = {opt: probs[tid].item() for opt, tid in option_ids.items()}
    return max(scores, key=scores.get)


def per_sample_metrics(
    logits: torch.Tensor,
    option_ids: Dict[str, int],
    expected: str,
) -> Dict[str, float]:
    """Record multiple metrics for a single sample so we can aggregate later.

    Returns argmax-correct, P(correct), logit(correct), and logit_diff =
    logit(correct) - max(logit(other options)). Logit-diff is the field-standard
    metric (Wang 2023, Zhang 2024); argmax-correct is what iter-04 used.
    """
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)

    option_logits = {opt: last_logits[tid].item() for opt, tid in option_ids.items()}
    option_probs = {opt: probs[tid].item() for opt, tid in option_ids.items()}

    if expected not in option_logits:
        # No valid expected answer token — skip this sample
        return None

    pred = max(option_logits, key=option_logits.get)
    correct_logit = option_logits[expected]
    correct_prob = option_probs[expected]
    other = [v for k, v in option_logits.items() if k != expected]
    max_other = max(other) if other else float("-inf")
    return {
        "expected": expected,
        "predicted": pred,
        "correct": int(pred == expected),
        "correct_logit": correct_logit,
        "correct_prob": correct_prob,
        "max_other_logit": max_other,
        "logit_diff": correct_logit - max_other,
    }


def cache_z_for_heads(
    model,
    tokens: torch.Tensor,
    target_layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Run forward and cache hook_z for the listed layers.

    Returns: {layer_idx: tensor of shape [seq_len, n_heads, head_dim] on CPU}
    """
    layer_set = set(target_layers)
    names_filter = [f"blocks.{l}.attn.hook_z" for l in target_layers]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names_filter)
    out = {}
    for l in target_layers:
        z = cache[f"blocks.{l}.attn.hook_z"]  # [batch, seq, head, head_dim]
        out[l] = z[0].detach().to("cpu")
    return out


def make_resample_hook(
    layer_idx: int,
    head_indices: List[int],
    donor_z: torch.Tensor,
):
    """Hook that replaces hook_z[:, :, head_indices, :] with donor activations.

    Right-aligned: donor's tail replaces test's tail. This matters because
    answer-option tokens and the END position sit at the tail of the sequence,
    and counterfactual corruption (placeholder substitutions like [V], [E])
    also concentrates there. The earlier left-aligned implementation was
    discarding the causally-meaningful donor tail.

    donor_z is on CPU shape [donor_seq_len, n_heads, head_dim].
    """
    def hook_fn(z, hook):
        seq_len = z.shape[1]
        donor_len = donor_z.shape[0]
        ml = min(seq_len, donor_len)
        # Take the last `ml` positions from donor; place at test's last `ml`.
        d = donor_z[-ml:].to(z.device).to(z.dtype)
        for h in head_indices:
            z[:, -ml:, h, :] = d[:, h, :]
        return z
    return hook_fn


def make_mean_hook(
    layer_idx: int,
    head_indices: List[int],
    mean_z: torch.Tensor,
):
    """Hook that replaces hook_z with the dataset-mean activation, right-aligned.

    mean_z: [max_len, n_heads, head_dim] on CPU, where max_len is the longest
    counterfactual sample in the precompute set; positions are right-padded with
    zeros (positions [0, max_len - this_sample_len) for short samples).

    Standard mean ablation per Wang 2023, Zhang 2024.
    """
    def hook_fn(z, hook):
        seq_len = z.shape[1]
        max_len = mean_z.shape[0]
        ml = min(seq_len, max_len)
        # Right-align: take last `ml` positions of mean_z, place at test's tail.
        d = mean_z[-ml:].to(z.device).to(z.dtype)
        for h in head_indices:
            z[:, -ml:, h, :] = d[:, h, :].unsqueeze(0)  # broadcast batch
        return z
    return hook_fn


def precompute_mean_z(
    model,
    tokenizer,
    samples: List[Dict],
    target_layers: List[int],
    device: str = "cuda",
    n_pool: int = 100,
) -> Dict[int, torch.Tensor]:
    """Precompute mean(hook_z) across counterfactual incorrect_prompts.

    Right-aligns sequences to the maximum length in the pool. Positions where
    a sample is shorter than max_len contribute 0; we divide by per-position
    count to get the actual mean over contributing samples.

    Returns: {layer_idx: tensor [max_len, n_heads, head_dim]} on CPU.

    n_pool caps the precompute set (default 100) for speed; this approximates
    the full dataset's mean with negligible loss at our K (5-10 heads).
    """
    pool = samples[:n_pool]
    print(f"[mean-precompute] caching {len(pool)} incorrect_prompts on layers {target_layers}")

    # First pass: discover max length
    lengths = []
    cached = []  # list of {layer: z[seq, head, dim]}
    for sample in tqdm(pool, desc="precompute mean"):
        i_tokens = tokenizer.encode(sample["incorrect_prompt"], return_tensors="pt").to(device)
        z_by_layer = cache_z_for_heads(model, i_tokens, target_layers)
        cached.append(z_by_layer)
        lengths.append(i_tokens.shape[1])
        del i_tokens
        torch.cuda.empty_cache() if device == "cuda" else None

    max_len = max(lengths)
    # Infer n_heads, head_dim from one cached layer
    any_z = next(iter(cached[0].values()))
    n_heads, head_dim = any_z.shape[1], any_z.shape[2]

    sums = {l: torch.zeros(max_len, n_heads, head_dim) for l in target_layers}
    counts = torch.zeros(max_len)
    for z_by_layer, seq_len in zip(cached, lengths):
        # Right-align this sample into the max_len buffer
        start = max_len - seq_len
        counts[start:] += 1
        for l in target_layers:
            sums[l][start:] += z_by_layer[l]

    means = {}
    safe_counts = counts.clamp(min=1).unsqueeze(-1).unsqueeze(-1)  # [max_len, 1, 1]
    for l in target_layers:
        means[l] = sums[l] / safe_counts
    return means


def _aggregate(per_sample: List[Dict]) -> Dict:
    """Aggregate per-sample metrics into summary stats."""
    valid = [s for s in per_sample if s is not None]
    n = len(valid)
    if n == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "mean_correct_prob": 0.0, "mean_logit_diff": 0.0,
                "mean_correct_logit": 0.0}
    return {
        "accuracy": sum(s["correct"] for s in valid) / n,
        "correct": sum(s["correct"] for s in valid),
        "total": n,
        "mean_correct_prob": sum(s["correct_prob"] for s in valid) / n,
        "mean_logit_diff": sum(s["logit_diff"] for s in valid) / n,
        "mean_correct_logit": sum(s["correct_logit"] for s in valid) / n,
    }


def evaluate_with_ablation(
    model,
    tokenizer,
    samples: List[Dict],
    target_heads: List[Tuple[int, int]],
    ablation: str = "resample",
    mean_z_cache: Optional[Dict[int, torch.Tensor]] = None,
    device: str = "cuda",
    desc: str = "eval",
) -> Dict:
    """Evaluate with knockout ablation of target_heads. Records all metrics
    (accuracy, correct_prob, logit_diff, correct_logit) so downstream can use
    whichever is field-standard for the comparison.

    ablation = "resample"  → per-sample donor from incorrect_prompt (iter-04)
    ablation = "mean"      → dataset-mean activation (Wang 2023, Zhang 2024)
                             requires mean_z_cache precomputed.
    """
    option_ids = get_answer_token_ids(tokenizer)
    heads_by_layer: Dict[int, List[int]] = {}
    for l, h in target_heads:
        heads_by_layer.setdefault(l, []).append(h)
    target_layers = sorted(heads_by_layer.keys())

    per_sample = []

    for sample in tqdm(samples, desc=desc):
        correct_prompt = sample["correct_prompt"]
        expected = sample["answer"]
        c_tokens = tokenizer.encode(correct_prompt, return_tensors="pt").to(device)

        # Build hooks based on ablation mode
        hooks = []
        donor_cache = None
        if ablation == "resample":
            incorrect_prompt = sample["incorrect_prompt"]
            i_tokens = tokenizer.encode(incorrect_prompt, return_tensors="pt").to(device)
            donor_cache = cache_z_for_heads(model, i_tokens, target_layers)
            for l, h_list in heads_by_layer.items():
                hooks.append((
                    f"blocks.{l}.attn.hook_z",
                    make_resample_hook(l, h_list, donor_cache[l]),
                ))
            del i_tokens
        elif ablation == "mean":
            assert mean_z_cache is not None, "mean_z_cache required for ablation=mean"
            for l, h_list in heads_by_layer.items():
                hooks.append((
                    f"blocks.{l}.attn.hook_z",
                    make_mean_hook(l, h_list, mean_z_cache[l]),
                ))
        else:
            raise ValueError(f"Unknown ablation: {ablation}")

        with torch.no_grad():
            logits = model.run_with_hooks(c_tokens, fwd_hooks=hooks)

        m = per_sample_metrics(logits, option_ids, expected)
        per_sample.append(m)

        del c_tokens, logits
        if donor_cache is not None:
            del donor_cache
        torch.cuda.empty_cache() if device == "cuda" else None

    agg = _aggregate(per_sample)
    agg["n_heads"] = len(target_heads)
    agg["heads"] = [list(h) for h in target_heads]
    agg["ablation"] = ablation
    return agg


def evaluate_baseline(
    model,
    tokenizer,
    samples: List[Dict],
    device: str = "cuda",
    desc: str = "baseline",
) -> Dict:
    """Baseline accuracy + correct_prob + logit_diff on correct_prompt (no ablation)."""
    option_ids = get_answer_token_ids(tokenizer)
    per_sample = []
    for sample in tqdm(samples, desc=desc):
        c_tokens = tokenizer.encode(sample["correct_prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(c_tokens)
        m = per_sample_metrics(logits, option_ids, sample["answer"])
        per_sample.append(m)
    return _aggregate(per_sample)


def run_cell(
    model_name: str,
    dataset_key: str,
    max_samples: int = 500,
    n_random_trials: int = N_RANDOM_TRIALS,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
    model=None,
    ablation: str = "mean",
    mean_pool_n: int = 100,
    random_control: str = "global",
) -> Dict:
    """Run one (model, dataset) cell. If `model` is passed, reuse it; otherwise load.

    ablation = "mean" (default, field-standard per Wang/Zhang) or "resample"
    """
    print("=" * 70)
    print(f"Knockout v2: model={model_name}, dataset={dataset_key}, ablation={ablation}")
    print(f"  N={max_samples}, random_trials={n_random_trials}, random_control={random_control}")
    print("=" * 70)

    # Fix W1: rng seed derives from dataset name so each dataset gets
    # independent random head sets (was: identical across datasets in iter-04).
    # Uses md5 (not builtin hash()) because Python randomizes str hash() per
    # process (PYTHONHASHSEED) — builtin hash would make dataset_seed, and thus
    # which random heads get sampled, non-reproducible across reruns/machines.
    # Note: this changes the RNG stream vs. earlier "global" control runs,
    # which is fine — those runs' own seed was already process-random, so this
    # doesn't break comparability with anything already reported.
    set_all_seeds(RANDOM_SEED)
    dataset_hash = int(hashlib.md5(dataset_key.encode("utf-8")).hexdigest(), 16)
    dataset_seed = RANDOM_SEED + dataset_hash % 100000

    # 1. Load NS heads (per-dataset) + union for exclusion
    ns_heads, ns_union = load_per_dataset_ns_heads(dataset_key, model_name)
    ensure_nonempty_head_set(ns_heads, dataset_key)
    print(f"[heads] Per-dataset NS for {dataset_key}: K={len(ns_heads)}")
    for l, h in ns_heads:
        print(f"  L{l}-H{h}")
    print(f"[heads] Union NS across all datasets (random excludes this): K={len(ns_union)}")

    # 2. Load samples
    samples = load_samples(dataset_key, max_samples=max_samples)
    print(f"[data] Using {len(samples)} samples")

    # 3. Load model (reuse if passed)
    if model is None:
        model = load_hooked_transformer(model_name, device=device)
    tokenizer = model.tokenizer
    n_layers = model.cfg.n_layers
    n_heads_per_layer = model.cfg.n_heads
    print(f"[model] {n_layers} layers, {n_heads_per_layer} heads/layer")

    K = len(ns_heads)

    # 4. Baseline (records accuracy, correct_prob, logit_diff)
    baseline_result = evaluate_baseline(model, tokenizer, samples, device=device,
                                        desc=f"baseline {dataset_key}")
    print(f"[baseline] acc={baseline_result['accuracy']:.4f}  "
          f"prob={baseline_result['mean_correct_prob']:.4f}  "
          f"logit_diff={baseline_result['mean_logit_diff']:+.4f}")

    # 5. Precompute mean_z cache if ablation=mean. Pool over union of NS + the
    # random control's max-K head set, since all of these will be ablated.
    all_layers_to_cache = set()
    for l, _ in ns_heads:
        all_layers_to_cache.add(l)
    # Reserve also layers that random trials may pick — but random can pick any
    # layer, so we'd need to cache everything. Compromise: cache only NS layers
    # for NS knockout; recompute per random trial (only on those layers).
    mean_z_cache_ns = None
    if ablation == "mean":
        ns_layers = sorted(set(l for l, _ in ns_heads))
        mean_z_cache_ns = precompute_mean_z(
            model, tokenizer, samples[:mean_pool_n], ns_layers,
            device=device, n_pool=mean_pool_n,
        )

    # 6. NS knockout
    ns_result = evaluate_with_ablation(
        model, tokenizer, samples, ns_heads,
        ablation=ablation, mean_z_cache=mean_z_cache_ns,
        device=device, desc=f"NS-{ablation} {dataset_key}",
    )
    print(f"[NS {ablation}] acc={ns_result['accuracy']:.4f}  "
          f"prob={ns_result['mean_correct_prob']:.4f}  "
          f"logit_diff={ns_result['mean_logit_diff']:+.4f}  "
          f"(Δacc={ns_result['accuracy']-baseline_result['accuracy']:+.4f}, "
          f"Δlogit_diff={ns_result['mean_logit_diff']-baseline_result['mean_logit_diff']:+.4f})")

    # 7. Random control × n_random_trials, independent rng per cell
    rng = random.Random(dataset_seed)
    random_trials = []
    for trial in range(n_random_trials):
        if random_control == "layer_matched":
            rh, n_fallback = get_layer_matched_random_heads_excluding(
                ns_heads, n_layers, n_heads_per_layer, ns_union, rng
            )
            if n_fallback > 0:
                print(f"  !!! WARNING: layer-matched control trial {trial+1} fell back to "
                      f"the global pool for {n_fallback}/{K} heads (an NS layer ran out of "
                      f"non-excluded heads) — control is no longer a clean per-layer match !!!")
        else:
            rh = get_random_heads_excluding(K, n_layers, n_heads_per_layer, ns_union, rng)
        # If ablation=mean, precompute mean_z over THIS trial's random head layers
        mean_z_cache_r = None
        if ablation == "mean":
            r_layers = sorted(set(l for l, _ in rh))
            mean_z_cache_r = precompute_mean_z(
                model, tokenizer, samples[:mean_pool_n], r_layers,
                device=device, n_pool=mean_pool_n,
            )
        r_result = evaluate_with_ablation(
            model, tokenizer, samples, rh,
            ablation=ablation, mean_z_cache=mean_z_cache_r,
            device=device, desc=f"random-{trial+1} {dataset_key}",
        )
        if random_control == "layer_matched":
            r_result["layer_match_fallback_n"] = n_fallback
        print(f"[random {trial+1}] acc={r_result['accuracy']:.4f}  "
              f"prob={r_result['mean_correct_prob']:.4f}  "
              f"logit_diff={r_result['mean_logit_diff']:+.4f}")
        random_trials.append(r_result)
        del mean_z_cache_r

    # Pooled random (accuracy + logit_diff)
    rand_acc_mean = float(np.mean([t["accuracy"] for t in random_trials]))
    rand_logit_diff_mean = float(np.mean([t["mean_logit_diff"] for t in random_trials]))
    rand_prob_mean = float(np.mean([t["mean_correct_prob"] for t in random_trials]))
    rand_correct_pooled = int(sum(t["correct"] for t in random_trials))
    rand_total_pooled = int(sum(t["total"] for t in random_trials))

    out = {
        "metadata": {
            "model": model_name,
            "dataset": dataset_key,
            "max_samples": max_samples,
            "actual_samples": len(samples),
            "n_random_trials": n_random_trials,
            "ablation": ablation,
            "random_control": random_control,
            "head_set_definition": "per_dataset_ns",
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "dataset_seed": dataset_seed,
            "mean_pool_n": mean_pool_n if ablation == "mean" else None,
        },
        "ns_heads": [list(h) for h in ns_heads],
        "ns_union_excluded_from_random": [list(h) for h in ns_union],
        "baseline": baseline_result,
        "ns_knockout": ns_result,
        "random_trials": random_trials,
        "random_pooled": {
            "mean_accuracy": rand_acc_mean,
            "mean_correct_prob": rand_prob_mean,
            "mean_logit_diff": rand_logit_diff_mean,
            "correct_pooled": rand_correct_pooled,
            "total_pooled": rand_total_pooled,
        },
    }

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / model_short_name(model_name) / "knockout"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # NOTE: existing consumers (summarize_iter05.py, threshold_sensitivity.py)
    # glob for "{dataset}_iter05_{ablation}_per_dataset_ns_*.json" to pick up
    # only the "global" random-control runs. A non-"global" control token must
    # NOT sit after "per_dataset_ns" (e.g. the old "..._per_dataset_ns_layer_matched_..."
    # scheme) or that glob's trailing "*" would silently swallow it too. Putting
    # the control token *before* "per_dataset_ns" keeps non-global runs out of
    # that pattern's match set — confirmed with fnmatch.fnmatch() against both
    # filename shapes before landing this change.
    if random_control == "global":
        out_path = output_dir / f"{dataset_key}_iter05_{ablation}_per_dataset_ns_{ts}.json"
    else:
        out_path = output_dir / f"{dataset_key}_iter05_{ablation}_{random_control}_per_dataset_ns_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_CONFIG.keys()), default="chinese_legal")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--n-random-trials", type=int, default=N_RANDOM_TRIALS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ablation", choices=["mean", "resample"], default="mean",
                   help="mean (Wang/Zhang field-standard) or resample (iter-04 ACDC-style)")
    p.add_argument("--mean-pool-n", type=int, default=100,
                   help="number of incorrect_prompts to average for mean ablation")
    p.add_argument("--random-control", choices=["global", "layer_matched"], default="global",
                   help="global samples any non-NS head; layer_matched preserves NS per-layer counts")
    args = p.parse_args()

    datasets = list(DATASET_CONFIG.keys()) if args.all_datasets else [args.dataset]

    # Load model once, reuse across all datasets
    model = load_hooked_transformer(args.model, device=args.device)
    print(f"[main] Model loaded; running {len(datasets)} dataset(s) with ablation={args.ablation}")

    run_dataset_batch(
        datasets,
        lambda ds: run_cell(
                model_name=args.model,
                dataset_key=ds,
                max_samples=args.max_samples,
                n_random_trials=args.n_random_trials,
                device=args.device,
                model=model,
                ablation=args.ablation,
                mean_pool_n=args.mean_pool_n,
                random_control=args.random_control,
            ),
    )


if __name__ == "__main__":
    main()
