
import glob
import json
import math
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS = [("Qwen3-8B", "qwen3-8b"), ("Llama-3.1-8B", "llama-3.1-8b"), ("Mistral-7B", "mistral-7b-v0.1")]
DATASETS = ["chinese_legal", "english_legal", "chinese_moral", "english_moral", "general_mcqa"]

def load_cell(model_short, dataset):
    pattern = str(PROJECT_ROOT / "output" / model_short / "knockout" / f"{dataset}_iter05_mean_per_dataset_ns_*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    with open(matches[-1]) as f:
        d = json.load(f)
    ns = d["ns_knockout"]
    trials_ac = [t["accuracy"] for t in d["random_trials"]]
    trials_ld = [t["mean_logit_diff"] for t in d["random_trials"]]
    m_ac = statistics.mean(trials_ac); s_ac = statistics.stdev(trials_ac)
    m_ld = statistics.mean(trials_ld); s_ld = statistics.stdev(trials_ld)
    se_ac = s_ac / math.sqrt(len(trials_ac))
    se_ld = s_ld / math.sqrt(len(trials_ld))
    z_ac = (ns["accuracy"] - m_ac) / se_ac if se_ac > 0 else 0
    z_ld = (ns["mean_logit_diff"] - m_ld) / se_ld if se_ld > 0 else 0
    return {
        "gap_acc": ns["accuracy"] - m_ac,
        "gap_ld": ns["mean_logit_diff"] - m_ld,
        "z_acc": z_ac,
        "z_ld": z_ld,
    }

cells = {}
for ml, ms in MODELS:
    for ds in DATASETS:
        c = load_cell(ms, ds)
        if c is not None:
            cells[(ml, ds)] = c

def count_sig(acc_thr, ld_thr, z_thr):
    n = 0
    for c in cells.values():
        acc_ok = (c["gap_acc"] <= -acc_thr) and (c["z_acc"] <= -z_thr)
        ld_ok  = (c["gap_ld"]  <= -ld_thr) and (c["z_ld"]  <= -z_thr)
        if acc_ok or ld_ok:
            n += 1
    return n

def count_sig_ld_only(ld_thr, z_thr):
    n = 0
    for c in cells.values():
        if (c["gap_ld"] <= -ld_thr) and (c["z_ld"] <= -z_thr):
            n += 1
    return n

ACC_THRS = [0.005, 0.01, 0.02, 0.03]
LD_THRS  = [0.03, 0.05, 0.10]
Z_THRS   = [1.5, 2.0, 2.5, 3.0]

print("=" * 90)
print("Threshold sensitivity: count of cells (out of 15) satisfying criterion")
print("Criterion: (|Δacc| ≥ acc_thr AND z_acc ≤ -z_thr) OR (|Δld| ≥ ld_thr AND z_ld ≤ -z_thr)")
print("=" * 90)
print()
for z_thr in Z_THRS:
    print(f"Trial z-cutoff = -{z_thr}")
    hdr = f"  {'acc_thr ↓ / ld_thr →':<25}"
    for ld in LD_THRS:
        hdr += f" {ld:>6}"
    print(hdr)
    print("  " + "-" * (25 + 7 * len(LD_THRS)))
    for ac in ACC_THRS:
        row = f"  {ac*100:>4.1f}pp" + " " * 19
        for ld in LD_THRS:
            n = count_sig(ac, ld, z_thr)
            row += f" {n:>6}"
        print(row)
    print()

print("=" * 90)
print(f"HEADLINE criterion (paper, |Δld|≥0.05, |z|≥2, logit-diff only): {count_sig_ld_only(0.05, 2.0)}/15")
print(f"OR-criterion (sensitivity ref, |Δacc|≥1pp OR |Δld|≥0.05, |z|≥2):   {count_sig(0.01, 0.05, 2.0)}/15")
print("=" * 90)

all_counts = [count_sig(ac, ld, z) for ac in ACC_THRS for ld in LD_THRS for z in Z_THRS]
print(f"Range across 4×3×4 = 48 threshold combinations: {min(all_counts)} – {max(all_counts)} of 15")
print(f"Median: {statistics.median(all_counts):.1f}, mean: {statistics.mean(all_counts):.2f}")
