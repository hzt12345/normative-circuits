#!/usr/bin/env python3
"""Validate and summarize the formal layer-matched rebuttal experiment."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


MODELS = {
    "Qwen/Qwen3-8B": "qwen3-8b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b",
    "mistralai/Mistral-7B-v0.1": "mistral-7b-v0.1",
}

DATASETS = (
    "chinese_legal",
    "english_legal",
    "chinese_moral",
    "english_moral",
    "general_mcqa",
)

NORMATIVE_DATASETS = frozenset(DATASETS[:-1])


class ResultValidationError(RuntimeError):
    pass


def _heads(value):
    return [tuple(head) for head in value]


def validate_result(
    result,
    expected_model,
    expected_dataset,
    expected_samples=500,
    expected_trials=5,
):
    """Reject any result that is not a complete, strict formal control cell."""
    metadata = result.get("metadata", {})
    expected = {
        "model": expected_model,
        "dataset": expected_dataset,
        "max_samples": expected_samples,
        "actual_samples": expected_samples,
        "n_random_trials": expected_trials,
        "ablation": "mean",
        "random_control": "layer_matched",
        "head_set_definition": "per_dataset_ns",
        "random_seed": 42,
        "mean_pool_n": 100,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ResultValidationError(
                f"{key}: expected {value!r}, got {metadata.get(key)!r}"
            )

    ns_heads = _heads(result.get("ns_heads", []))
    ns_union = set(_heads(result.get("ns_union_excluded_from_random", [])))
    if not ns_heads:
        raise ResultValidationError("ns_heads is empty")
    if not set(ns_heads).issubset(ns_union):
        raise ResultValidationError("NS union does not contain the cell's NS heads")

    trials = result.get("random_trials", [])
    if len(trials) != expected_trials:
        raise ResultValidationError(
            f"expected {expected_trials} random trials, got {len(trials)}"
        )

    aggregates = [result.get("baseline", {}), result.get("ns_knockout", {}), *trials]
    if any(item.get("total") != expected_samples for item in aggregates):
        raise ResultValidationError(
            f"all aggregates must contain {expected_samples} evaluated samples"
        )

    if result.get("ns_knockout", {}).get("n_heads") != len(ns_heads):
        raise ResultValidationError("NS knockout head count does not match ns_heads")
    ns_knockout_heads = _heads(result.get("ns_knockout", {}).get("heads", []))
    if len(ns_knockout_heads) != len(ns_heads) or set(ns_knockout_heads) != set(ns_heads):
        raise ResultValidationError("NS knockout heads do not match ns_heads")

    ns_layer_counts = Counter(layer for layer, _ in ns_heads)
    for index, trial in enumerate(trials, start=1):
        if trial.get("layer_match_fallback_n") != 0:
            raise ResultValidationError(
                f"layer-matched fallback detected in random trial {index}"
            )
        trial_heads = _heads(trial.get("heads", []))
        if trial.get("n_heads") != len(ns_heads):
            raise ResultValidationError(
                f"random trial {index} n_heads does not match NS head count"
            )
        if len(trial_heads) != len(ns_heads) or len(set(trial_heads)) != len(ns_heads):
            raise ResultValidationError(
                f"random trial {index} does not contain {len(ns_heads)} unique heads"
            )
        overlap = set(trial_heads) & ns_union
        if overlap:
            raise ResultValidationError(
                f"random trial {index} contains excluded NS head(s): {sorted(overlap)}"
            )
        trial_layer_counts = Counter(layer for layer, _ in trial_heads)
        if trial_layer_counts != ns_layer_counts:
            raise ResultValidationError(
                f"random trial {index} per-layer counts {dict(trial_layer_counts)} "
                f"do not match NS counts {dict(ns_layer_counts)}"
            )


def collect_formal_results(root, models=None, datasets=None, min_mtime=None):
    """Return the newest valid formal file for each expected model/dataset."""
    root = Path(root)
    models = MODELS if models is None else models
    datasets = DATASETS if datasets is None else tuple(datasets)
    collected = {}
    missing = []

    for model, model_short in models.items():
        for dataset in datasets:
            pattern = (
                root
                / "output"
                / model_short
                / "knockout"
                / f"{dataset}_iter05_mean_layer_matched_per_dataset_ns_*.json"
            )
            candidates = sorted(
                pattern.parent.glob(pattern.name),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            reasons = []
            for path in candidates:
                if min_mtime is not None and path.stat().st_mtime < min_mtime:
                    reasons.append(
                        f"{path.name}: predates run start ({path.stat().st_mtime} < "
                        f"{min_mtime})"
                    )
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    validate_result(result, model, dataset)
                except (OSError, json.JSONDecodeError, ResultValidationError) as exc:
                    reasons.append(f"{path.name}: {exc}")
                    continue
                collected[(model, dataset)] = (path, result)
                break
            else:
                detail = "; ".join(reasons) if reasons else "no candidate files"
                missing.append(f"{model}/{dataset} missing or invalid ({detail})")

    if missing:
        raise ResultValidationError("; ".join(missing))
    return collected


def _trial_test(ns_value, trial_values):
    mean = statistics.mean(trial_values)
    stdev = statistics.stdev(trial_values)
    if stdev == 0:
        if ns_value == mean:
            return mean, stdev, 0.0, 1.0
        return mean, stdev, math.copysign(math.inf, ns_value - mean), 0.0
    t_value = (ns_value - mean) / (stdev / math.sqrt(len(trial_values)))
    try:
        from scipy import stats

        p_value = float(2 * stats.t.cdf(-abs(t_value), df=len(trial_values) - 1))
    except ImportError:
        p_value = None
    return mean, stdev, t_value, p_value


def summarize_result(result):
    metadata = result["metadata"]
    baseline = result["baseline"]
    ns = result["ns_knockout"]
    trials = result["random_trials"]
    acc_values = [trial["accuracy"] for trial in trials]
    ld_values = [trial["mean_logit_diff"] for trial in trials]
    acc_mean, acc_sd, acc_t, acc_p = _trial_test(ns["accuracy"], acc_values)
    ld_mean, ld_sd, ld_t, ld_p = _trial_test(ns["mean_logit_diff"], ld_values)

    return {
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "normative": metadata["dataset"] in NORMATIVE_DATASETS,
        "n_heads": ns["n_heads"],
        "baseline_accuracy": baseline["accuracy"],
        "ns_accuracy": ns["accuracy"],
        "random_accuracy_mean": acc_mean,
        "random_accuracy_sd": acc_sd,
        "accuracy_gap": ns["accuracy"] - acc_mean,
        "accuracy_t": acc_t,
        "accuracy_p": acc_p,
        "baseline_logit_diff": baseline["mean_logit_diff"],
        "ns_logit_diff": ns["mean_logit_diff"],
        "random_logit_diff_mean": ld_mean,
        "random_logit_diff_sd": ld_sd,
        "logit_diff_gap": ns["mean_logit_diff"] - ld_mean,
        "logit_diff_t": ld_t,
        "logit_diff_p": ld_p,
        "accuracy_field_signal": (
            ns["accuracy"] - acc_mean <= -0.01 and acc_t <= -2
        ),
        "logit_diff_field_signal": (
            ns["mean_logit_diff"] - ld_mean <= -0.05 and ld_t <= -2
        ),
    }


def build_summary(collected):
    rows = []
    for (model, dataset), (path, result) in collected.items():
        row = summarize_result(result)
        row["source"] = str(path)
        rows.append(row)
    rows.sort(key=lambda row: (list(MODELS).index(row["model"]), DATASETS.index(row["dataset"])))

    normative = [row for row in rows if row["normative"]]
    fallbacks = sum(
        trial.get("layer_match_fallback_n", 0)
        for _, result in collected.values()
        for trial in result.get("random_trials", [])
    )
    return {
        "validation": {
            "valid_cells": len(rows),
            "expected_cells": len(MODELS) * len(DATASETS),
            "fallbacks": fallbacks,
        },
        "normative": {
            "cells": len(normative),
            "accuracy_gap_negative": sum(row["accuracy_gap"] < 0 for row in normative),
            "logit_diff_gap_negative": sum(
                row["logit_diff_gap"] < 0 for row in normative
            ),
            "field_signal_either": sum(
                row["accuracy_field_signal"] or row["logit_diff_field_signal"]
                for row in normative
            ),
            "mean_accuracy_gap": statistics.mean(
                row["accuracy_gap"] for row in normative
            ),
            "mean_logit_diff_gap": statistics.mean(
                row["logit_diff_gap"] for row in normative
            ),
        },
        "cells": rows,
    }


def _clean_json(value):
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_summary(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "layer_matched_summary.json"
    csv_path = output_dir / "layer_matched_summary.csv"
    md_path = output_dir / "layer_matched_summary.md"
    json_path.write_text(
        json.dumps(_clean_json(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = summary["cells"]
    clean_rows = _clean_json(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean_rows[0]))
        writer.writeheader()
        writer.writerows(clean_rows)

    lines = [
        "# Layer-Matched Random Control Summary",
        "",
        f"Validated cells: {summary['validation']['valid_cells']}/"
        f"{summary['validation']['expected_cells']}; "
        f"fallbacks: {summary['validation']['fallbacks']}",
        "",
        "| Model | Dataset | K | Base acc | NS acc | Random acc | Gap | "
        "Base LD | NS LD | Random LD | LD gap | Field signal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        signal = row["accuracy_field_signal"] or row["logit_diff_field_signal"]
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['n_heads']} | "
            f"{row['baseline_accuracy']:.3f} | {row['ns_accuracy']:.3f} | "
            f"{row['random_accuracy_mean']:.3f} | {row['accuracy_gap']:+.3f} | "
            f"{row['baseline_logit_diff']:+.3f} | {row['ns_logit_diff']:+.3f} | "
            f"{row['random_logit_diff_mean']:+.3f} | "
            f"{row['logit_diff_gap']:+.3f} | {'yes' if signal else 'no'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--min-mtime", type=float)
    args = parser.parse_args()

    collected = collect_formal_results(args.root, min_mtime=args.min_mtime)
    summary = build_summary(collected)
    summary["validation"]["min_mtime"] = args.min_mtime
    paths = write_summary(summary, args.root / "output")
    validation = summary["validation"]
    print(
        f"{validation['valid_cells']}/{validation['expected_cells']} valid cells; "
        f"{validation['fallbacks']} fallbacks; summaries: {', '.join(map(str, paths))}"
    )


if __name__ == "__main__":
    main()
