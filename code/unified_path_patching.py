

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

DATASET_CONFIG = {
    "chinese_legal": {
        "file_pattern": "data/processed/chinese_legal.json",
        "fallback_file": "data/processed/chinese_legal.json",
        "name": "Chinese Legal",
        "language": "zh",
        "domain": "legal",
        "count": 1999,
    },
    "english_legal": {
        "file_pattern": "data/processed/english_legal.json",
        "fallback_file": "data/processed/english_legal.json",
        "name": "English Legal",
        "language": "en",
        "domain": "legal",
        "count": 619,
    },
    "chinese_moral": {
        "file_pattern": "data/processed/chinese_moral.json",
        "fallback_file": "data/processed/chinese_moral.json",
        "name": "Chinese Moral",
        "language": "zh",
        "domain": "moral",
        "count": 1934,
    },
    "english_moral": {
        "file_pattern": "data/processed/english_moral.json",
        "fallback_file": "data/processed/english_moral.json",
        "name": "English Moral",
        "language": "en",
        "domain": "moral",
        "count": 1367,
    },
    "general_mcqa": {
        "file_pattern": "data/processed/general_mcqa_stem.json",
        "fallback_file": "data/processed/general_mcqa_stem.json",
        "name": "General MCQA (STEM)",
        "language": "en",
        "domain": "general",
        "count": 500,
    },
    "general_mcqa_humanities": {
        "file_pattern": "data/processed/general_mcqa_humanities.json",
        "fallback_file": "data/processed/general_mcqa_humanities.json",
        "name": "General MCQA (Humanities)",
        "language": "en",
        "domain": "general",
        "count": 500,
    },
}

SUPPORTED_MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-2-9b",
    "microsoft/phi-4",
]

DEFAULT_MODEL = "Qwen/Qwen3-8B"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Path Patching Experiment (Multi-Model Support)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}). Supported: {', '.join(SUPPORTED_MODELS)}",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()) + ["all"],
        default="chinese_legal",
        help="Dataset to process",
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["exact", "atp"],
        default="atp",
        help="Method: 'exact' (full path patching, slow) or 'atp' (attribution patching, fast)",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to process (for testing)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (only 1 supported)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: output/<model-short-name>/)",
    )

    parser.add_argument(
        "--model-cache",
        type=str,
        default="Model",
        help="Model cache directory",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save interval (every N samples)",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Direct data file path (overrides dataset config lookup)",
    )

    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Output subdirectory (e.g. 'sensitivity')",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (overrides dataset_key)",
    )

    return parser.parse_args()

def get_model_short_name(model_name: str) -> str:

    return model_name.split("/")[-1].lower()

def load_model(model_name: str, cache_dir: str = "Model", device: str = "cuda"):
    import torch
    from transformer_lens import HookedTransformer

    print("=" * 60)
    print(f"Loading model: {model_name}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Device: {device}")
    print("=" * 60)

    if device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA unavailable; switching to CPU")
            device = "cpu"
        else:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    from local_model_loader import load_hooked_transformer
    model = load_hooked_transformer(model_name, device=device)

    print(f"Model loaded.")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Attention heads: {model.cfg.n_heads}")
    print(f"  Hidden dim: {model.cfg.d_model}")
    print("=" * 60)

    return model

def find_dataset_file(dataset_key: str) -> Path:
    import glob
    config = DATASET_CONFIG[dataset_key]
    pattern = PROJECT_ROOT / config["file_pattern"]
    matches = sorted(glob.glob(str(pattern)), key=os.path.getmtime, reverse=True)

    if matches:
        return Path(matches[0])

    fallback = PROJECT_ROOT / config["fallback_file"]
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Dataset file for {dataset_key} not found: {pattern}")

def load_dataset(dataset_key: str, data_file: str = None) -> List[Dict]:
    config = DATASET_CONFIG[dataset_key]

    if data_file:
        file_path = Path(data_file)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
    else:
        file_path = find_dataset_file(dataset_key)

    print(f"Loading dataset: {config['name']}")
    print(f"  File: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Loaded {len(data)} items")
    return data

def align_tokens(model, correct_tokens, incorrect_tokens, device):
    import torch
    target_len = correct_tokens.shape[1]
    current_len = incorrect_tokens.shape[1]

    pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id

    if current_len == target_len:
        return incorrect_tokens

    if current_len < target_len:

        padding = torch.full(
            (incorrect_tokens.shape[0], target_len - current_len),
            pad_id,
            dtype=incorrect_tokens.dtype,
            device=device,
        )

        return torch.cat([incorrect_tokens, padding], dim=1)

    elif current_len > target_len:

        return incorrect_tokens[:, :target_len]

def run_attribution_patching(
    model,
    correct_prompt: str,
    incorrect_prompt: str,
    answer: str,
) -> Dict:
    import torch

    correct_tokens = model.to_tokens(correct_prompt)
    incorrect_tokens = model.to_tokens(incorrect_prompt)
    incorrect_tokens = align_tokens(model, correct_tokens, incorrect_tokens, correct_tokens.device)
    answer_token = model.to_tokens(answer, prepend_bos=False)[0, 0]

    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(
            incorrect_tokens,
            names_filter=lambda name: name.endswith("hook_z")
        )

    model.reset_hooks()
    clean_cache = {}

    def save_act_hook(act, hook):
        act.retain_grad()
        clean_cache[hook.name] = act
        return act

    fwd_hooks = [(f"blocks.{l}.attn.hook_z", save_act_hook) for l in range(model.cfg.n_layers)]

    logits = model.run_with_hooks(
        correct_tokens,
        fwd_hooks=fwd_hooks
    )

    target_logit = logits[0, -1, answer_token]
    target_logit.backward()

    results = {}
    for layer in range(model.cfg.n_layers):
        layer_results = {}
        hook_name = f"blocks.{layer}.attn.hook_z"

        clean_z = clean_cache[hook_name]
        corrupted_z = corrupted_cache[hook_name]
        grad_z = clean_z.grad

        if grad_z is None:

            for head in range(model.cfg.n_heads):
                layer_results[f"head_{head}"] = {"prob_diff": 0.0, "method": "atp"}
        else:
            act_diff = clean_z.detach() - corrupted_z

            attribution = (act_diff * grad_z.detach()).sum(dim=(0, 1, 3))

            for head in range(model.cfg.n_heads):
                layer_results[f"head_{head}"] = {
                    "prob_diff": attribution[head].item(),
                    "method": "atp"
                }
        results[f"layer_{layer}"] = layer_results

    model.reset_hooks()
    del clean_cache, corrupted_cache
    torch.cuda.empty_cache()

    return results

def run_exact_patching(
    model,
    correct_prompt: str,
    incorrect_prompt: str,
    answer: str,
) -> Dict:
    import torch

    correct_tokens = model.to_tokens(correct_prompt)
    incorrect_tokens = model.to_tokens(incorrect_prompt)
    incorrect_tokens = align_tokens(model, correct_tokens, incorrect_tokens, correct_tokens.device)

    with torch.no_grad():
        correct_logits, correct_cache = model.run_with_cache(correct_tokens)
        incorrect_logits, incorrect_cache = model.run_with_cache(incorrect_tokens)

    answer_token = model.to_tokens(answer, prepend_bos=False)[0, 0]
    original_prob = torch.softmax(correct_logits[0, -1], dim=-1)[answer_token].detach()

    results = {}

    for layer in range(model.cfg.n_layers):
        layer_results = {}
        for head in range(model.cfg.n_heads):
            def patching_hook(activation, hook, head_idx=head):
                activation[:, :, head_idx, :] = incorrect_cache[hook.name][:, :, head_idx, :]
                return activation

            hook_name = f"blocks.{layer}.attn.hook_z"
            patched_logits = model.run_with_hooks(
                correct_tokens,
                fwd_hooks=[(hook_name, patching_hook)],
            )

            patched_prob = torch.softmax(patched_logits[0, -1], dim=-1)[answer_token].detach()
            prob_diff = (patched_prob - original_prob).cpu().numpy()

            layer_results[f"head_{head}"] = {
                "original_prob": float(original_prob.cpu().numpy()),
                "patched_prob": float(patched_prob.cpu().numpy()),
                "prob_diff": float(prob_diff),
            }

        results[f"layer_{layer}"] = layer_results

    del correct_cache, incorrect_cache, correct_logits, incorrect_logits
    torch.cuda.empty_cache()

    return results

def analyze_dataset(
    model,
    data: List[Dict],
    method: str = "atp",
    max_samples: Optional[int] = None,
    save_interval: int = 50,
    output_file: Optional[Path] = None,
    existing_results: Optional[List[Dict]] = None,
) -> List[Dict]:
    import torch
    from tqdm import tqdm

    if max_samples:
        data = data[:max_samples]

    results = existing_results if existing_results else []
    start_idx = len(results)

    if start_idx > 0:
        print(f"Resuming: completed {start_idx}/{len(data)} samples")

    print(f"Using method: {method.upper()}")

    for idx, sample in enumerate(tqdm(data[start_idx:], initial=start_idx, total=len(data), desc=f"Analyzing ({method})")):
        actual_idx = idx + start_idx

        try:
            if method == "atp":
                patching_result = run_attribution_patching(
                    model,
                    sample["correct_prompt"],
                    sample["incorrect_prompt"],
                    sample["answer"],
                )
            else:
                patching_result = run_exact_patching(
                    model,
                    sample["correct_prompt"],
                    sample["incorrect_prompt"],
                    sample["answer"],
                )

            results.append({
                "sample": sample,
                "patching_result": patching_result,
                "index": actual_idx,
            })

            if output_file and (actual_idx + 1) % save_interval == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nIntermediate results saved: {len(results)} items")

        except Exception as e:
            print(f"\nError processing sample {actual_idx}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                "sample": sample,
                "patching_result": None,
                "index": actual_idx,
                "error": str(e),
            })

            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        if (actual_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return results

def get_output_dir(args) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        model_dir = get_model_short_name(args.model)
        output_dir = PROJECT_ROOT / "output" / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_checkpoint_file(output_dir: Path, dataset_key: str) -> Path:
    return output_dir / f"{dataset_key}_checkpoint.json"

def get_result_file(output_dir: Path, dataset_key: str, method: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{dataset_key}_{method}_results_{timestamp}.json"

def save_results(results: List[Dict], output_file: Path, metadata: Dict):
    output_data = {
        "metadata": metadata,
        "results": results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {output_file}")

def load_checkpoint(checkpoint_file: Path) -> Tuple[List[Dict], int]:
    if not checkpoint_file.exists():
        return [], 0
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, len(data)
    elif isinstance(data, dict) and "results" in data:
        return data["results"], len(data["results"])
    return [], 0

def run_single_dataset(args, model, dataset_key: str, data_file: str = None, output_prefix: str = None):
    config = DATASET_CONFIG[dataset_key]
    output_dir = get_output_dir(args)

    if hasattr(args, 'output_subdir') and args.output_subdir:
        output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

    file_key = output_prefix or dataset_key

    print("\n" + "=" * 60)
    print(f"Starting: {config['name']}" + (f" [{output_prefix}]" if output_prefix else ""))
    print(f"  Language: {config['language']}")
    print(f"  Domain: {config['domain']}")
    print(f"  Expected sample count: {config['count']}")
    if data_file:
        print(f"  Data file override: {data_file}")
    print("=" * 60)

    data = load_dataset(dataset_key, data_file=data_file)
    checkpoint_file = get_checkpoint_file(output_dir, file_key)
    existing_results = None

    if args.resume and checkpoint_file.exists():
        existing_results, count = load_checkpoint(checkpoint_file)
        print(f"Resuming: completed {count} / {len(data)} samples")

    results = analyze_dataset(
        model,
        data,
        method=args.method,
        max_samples=args.max_samples,
        save_interval=args.save_interval,
        output_file=checkpoint_file,
        existing_results=existing_results,
    )

    result_file = get_result_file(output_dir, file_key, args.method)
    metadata = {
        "dataset": dataset_key,
        "config": config,
        "model": args.model,
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "successful_samples": sum(1 for r in results if r.get("patching_result") is not None),
    }
    if data_file:
        metadata["data_file"] = str(data_file)
    if output_prefix:
        metadata["output_prefix"] = output_prefix

    save_results(results, result_file, metadata)

    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Checkpoint cleared: {checkpoint_file}")

    return results

def main():
    args = parse_args()

    print("=" * 60)
    print("Unified Path Patching Experiment")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method.upper()}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Save interval: {args.save_interval}")
    print(f"Resume: {'yes' if args.resume else 'no'}")
    print("=" * 60)

    model = load_model(args.model, cache_dir=args.model_cache, device=args.device)

    if args.dataset == "all":
        datasets_to_process = list(DATASET_CONFIG.keys())
    else:
        datasets_to_process = [args.dataset]

    all_results = {}
    for dataset_key in datasets_to_process:
        results = run_single_dataset(
            args, model, dataset_key,
            data_file=args.data_file,
            output_prefix=args.output_prefix,
        )
        all_results[dataset_key] = results

    print("\n" + "=" * 60)
    print("Experiment summary")
    print("=" * 60)
    for dataset_key, results in all_results.items():
        config = DATASET_CONFIG[dataset_key]
        successful = sum(1 for r in results if r.get("patching_result") is not None)
        print(f"  {config['name']}: {successful}/{len(results)} samples succeeded")
    print("=" * 60)

if __name__ == "__main__":
    main()
