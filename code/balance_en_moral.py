"""
Balance EN Moral dataset by randomly swapping A/B option order in half the samples.

Original english_moral_1367_*.json: 100% A (all 1367 samples have answer "A",
i.e. the moral action is always listed as option A). This makes "accuracy" on
EN Moral degenerate to "model's preference for token A", not actual moral
judgment ability.

Fix: for a deterministic random 50% of samples, swap A and B in:
  - correct_prompt (rewrite "A: <moral>\nB: <immoral>" -> "A: <immoral>\nB: <moral>")
  - incorrect_prompt (same swap)
  - answer ("A" -> "B")

The scenarios stay identical; only option order changes.
"""
import json
import re
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
SRC = PROJECT_ROOT / "data" / "processed" / "english_moral_unbalanced.json"
DST = PROJECT_ROOT / "data" / "processed" / "english_moral.json"
SEED = 20260511


def swap_ab_in_prompt(prompt: str) -> str:
    """Swap content of "A: <text>\nB: <text>" lines in a prompt.

    Uses a regex that matches the two-line block at the end of the prompt.
    """
    # The prompts end with:
    #   ...\n\nA: <action_a>\nB: <action_b>
    # We swap the two action strings, keeping "A:" / "B:" labels.
    pattern = re.compile(r"(A:\s*)(.+?)(\nB:\s*)(.+?)(\n*\Z)", re.DOTALL)
    m = pattern.search(prompt)
    if not m:
        raise ValueError(f"Could not find A:/B: block in prompt:\n{prompt[-300:]}")
    prefix = prompt[: m.start()]
    label_a, action_a, label_b, action_b, suffix = m.groups()
    swapped = f"{prefix}{label_a}{action_b}{label_b}{action_a}{suffix}"
    return swapped


def main():
    with open(SRC) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {SRC.name}")
    print(f"Original answer distribution: {dict(Counter(s['answer'] for s in samples))}")

    rng = random.Random(SEED)
    swap_flags = [rng.random() < 0.5 for _ in samples]
    print(f"Will swap {sum(swap_flags)}/{len(samples)} samples (seed {SEED})")

    balanced = []
    for s, swap in zip(samples, swap_flags):
        if not swap:
            balanced.append(s)
            continue
        new = dict(s)
        new["correct_prompt"] = swap_ab_in_prompt(s["correct_prompt"])
        new["incorrect_prompt"] = swap_ab_in_prompt(s["incorrect_prompt"])
        new["answer"] = "B" if s["answer"] == "A" else "A"
        new_meta = dict(s.get("metadata", {}))
        new_meta["ab_swapped"] = True
        new["metadata"] = new_meta
        balanced.append(new)

    print(f"Balanced answer distribution: {dict(Counter(s['answer'] for s in balanced))}")

    # Sanity-check: pick one swapped sample and verify it makes sense
    for i, swap in enumerate(swap_flags):
        if swap:
            print("\n=== Sanity check: sample index", i, "===")
            print("ORIGINAL (last 300 chars):")
            print(samples[i]["correct_prompt"][-300:])
            print("BALANCED (last 300 chars):")
            print(balanced[i]["correct_prompt"][-300:])
            print(f"answer: {samples[i]['answer']} -> {balanced[i]['answer']}")
            break

    with open(DST, "w") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)
    print(f"\nSaved balanced dataset to {DST.name}")


if __name__ == "__main__":
    main()
