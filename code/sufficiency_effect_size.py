

DATA = {

    ("Qwen3-8B", "CN Legal"): (18, [10, 6, 11, 9, 7]),
    ("Qwen3-8B", "CN Moral"): (17, [13, 4, 3, 11, 7]),
    ("Qwen3-8B", "EN Legal"): (2,  [1, 1, 0, 2, 1]),
    ("Qwen3-8B", "EN Moral"): (13, [0, 2, 3, 4, 3]),

    ("Llama-3.1-8B", "CN Legal"): (9, [3, 3, 4, 4, 3]),
    ("Llama-3.1-8B", "CN Moral"): (3, [0, 1, 1, 1, 2]),
    ("Llama-3.1-8B", "EN Legal"): (6, [4, 0, 2, 3, 2]),
    ("Llama-3.1-8B", "EN Moral"): (0, [0, 0, 0, 0, 0]),

    ("Mistral-7B", "CN Legal"): (5, [4, 1, 3, 4, 2]),
    ("Mistral-7B", "CN Moral"): (6, [2, 5, 3, 1, 4]),
    ("Mistral-7B", "EN Legal"): (3, [4, 1, 3, 2, 2]),
    ("Mistral-7B", "EN Moral"): (14, [7, 3, 3, 6, 2]),
}

import statistics
import math

def cdf(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))

print("=" * 95)
print("Sufficiency under matched effect-size criterion:")
print("  gap = NS recovery − random recovery (out of N=100 incorrect samples per cell)")
print("  Significant: gap ≥ +1pp AND trial-level z ≥ +2")
print("=" * 95)
print()
hdr = f"{'Cell':<28}{'NS':>5}{'Rand μ':>8}{'Rand σ':>8}{'gap (pp)':>10}{'z (trial)':>10}{'Sig?':>6}"
print(hdr)
print("-" * 95)

n_total = 0
n_sig = 0
n_directional = 0
for (model, ds), (ns_n, trials) in DATA.items():
    rand_mean = statistics.mean(trials)
    rand_std = statistics.stdev(trials) if len(set(trials)) > 1 else 0.0
    rand_se = rand_std / math.sqrt(len(trials)) if rand_std > 0 else 0.0
    gap = ns_n - rand_mean
    z = gap / rand_se if rand_se > 0 else float("inf") if gap > 0 else 0
    sig = gap >= 1.0 and z >= 2.0
    directional = gap > 0
    print(f"{model+'/'+ds:<28}{ns_n:>5}{rand_mean:>8.2f}{rand_std:>8.2f}{gap:>+10.2f}{z:>+10.2f}{'✓' if sig else '':>6}")
    n_total += 1
    if sig: n_sig += 1
    if directional: n_directional += 1

print("-" * 95)
print(f"\nCounts (out of {n_total} cells):")
print(f"  Sig under matched effect-size (gap ≥ 1pp AND z ≥ 2): {n_sig}/{n_total}")
print(f"  Directional (gap > 0, sign-test):                    {n_directional}/{n_total}")
print(f"\nSign-test for directional: 2-sided p = {2 * cdf(-(n_directional - n_total/2) / math.sqrt(n_total/4)):.4f}")
