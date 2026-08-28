

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from transformer_lens import HookedTransformer
import logging

logger = logging.getLogger(__name__)

class AttentionHeadSuppressor:

    def __init__(self,
                 model: HookedTransformer,
                 suppression_strength: float = 0.5,
                 suppression_strategy: str = "multiply"):
        self.model = model
        self.suppression_strength = suppression_strength
        self.suppression_strategy = suppression_strategy
        self.target_heads = []
        self.hooks = []

    def set_target_heads(self, target_heads: List[Tuple[int, int]]):
        self.target_heads = target_heads
        logger.info(f"Set target heads for suppression: {target_heads}")

    def create_suppression_hook(self, layer_idx: int, head_indices: List[int]) -> Callable:
        def suppression_hook(activation: torch.Tensor, hook) -> torch.Tensor:

            modified_activation = activation.clone()

            for head_idx in head_indices:
                if head_idx < activation.shape[2]:
                    if self.suppression_strategy == "multiply":

                        modified_activation[:, :, head_idx, :] *= (1.0 - self.suppression_strength)

                    elif self.suppression_strategy == "zero":

                        modified_activation[:, :, head_idx, :] = 0.0

                    elif self.suppression_strategy == "noise":

                        noise_scale = self.suppression_strength * torch.std(activation[:, :, head_idx, :])
                        noise = torch.randn_like(activation[:, :, head_idx, :]) * noise_scale
                        modified_activation[:, :, head_idx, :] += noise

                    elif self.suppression_strategy == "scale":

                        modified_activation[:, :, head_idx, :] *= self.suppression_strength

                    else:
                        raise ValueError(f"Unknown suppression strategy: {self.suppression_strategy}")

            return modified_activation

        return suppression_hook

    def setup_hooks(self) -> List[Tuple[str, Callable]]:
        hooks = []

        layer_heads = {}
        for layer_idx, head_idx in self.target_heads:
            if layer_idx not in layer_heads:
                layer_heads[layer_idx] = []
            layer_heads[layer_idx].append(head_idx)

        for layer_idx, head_indices in layer_heads.items():
            hook_name = f"blocks.{layer_idx}.attn.hook_z"
            hook_func = self.create_suppression_hook(layer_idx, head_indices)
            hooks.append((hook_name, hook_func))

        self.hooks = hooks
        logger.info(f"Created {len(hooks)} suppression hooks")

        return hooks

    def run_with_suppression(self,
                           tokens: torch.Tensor,
                           return_type: str = "logits") -> torch.Tensor:
        if not self.hooks:
            self.setup_hooks()

        if return_type == "logits":
            return self.model.run_with_hooks(tokens, fwd_hooks=self.hooks)
        elif return_type == "cache":
            _, cache = self.model.run_with_cache(tokens, fwd_hooks=self.hooks)
            return cache
        elif return_type == "both":
            return self.model.run_with_cache(tokens, fwd_hooks=self.hooks)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    def compare_outputs(self,
                       tokens: torch.Tensor,
                       answer_token: torch.Tensor) -> Dict[str, Any]:

        with torch.no_grad():
            baseline_logits = self.model(tokens)
            baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)
            baseline_answer_prob = baseline_probs[answer_token].item()

        with torch.no_grad():
            suppressed_logits = self.run_with_suppression(tokens)
            suppressed_probs = torch.softmax(suppressed_logits[0, -1], dim=-1)
            suppressed_answer_prob = suppressed_probs[answer_token].item()

        prob_diff = suppressed_answer_prob - baseline_answer_prob
        prob_ratio = suppressed_answer_prob / baseline_answer_prob if baseline_answer_prob > 0 else 0

        return {
            "baseline_prob": baseline_answer_prob,
            "suppressed_prob": suppressed_answer_prob,
            "prob_diff": prob_diff,
            "prob_ratio": prob_ratio,
            "suppression_strength": self.suppression_strength,
            "suppression_strategy": self.suppression_strategy,
            "target_heads": self.target_heads
        }

class MultiStrengthSuppressor:

    def __init__(self,
                 model: HookedTransformer,
                 target_heads: List[Tuple[int, int]],
                 suppression_strategy: str = "multiply"):
        self.model = model
        self.target_heads = target_heads
        self.suppression_strategy = suppression_strategy

    def run_experiment(self,
                      tokens: torch.Tensor,
                      answer_token: torch.Tensor,
                      suppression_strengths: List[float] = [0.0, 0.2, 0.5, 0.8, 1.0]) -> Dict[str, Any]:
        results = {}

        for strength in suppression_strengths:
            suppressor = AttentionHeadSuppressor(
                model=self.model,
                suppression_strength=strength,
                suppression_strategy=self.suppression_strategy
            )
            suppressor.set_target_heads(self.target_heads)

            result = suppressor.compare_outputs(tokens, answer_token)
            results[f"strength_{strength}"] = result

        return results

def load_top_heads(file_path: str = "output/02-Moral-Analysis/top_legal_heads.json") -> List[Tuple[int, int]]:
    import json

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        top_heads = data.get("top_heads", [])
        logger.info(f"Loaded {len(top_heads)} top heads from {file_path}")

        return top_heads

    except Exception as e:
        logger.error(f"Failed to load top heads from {file_path}: {e}")

        return [(35, 24), (35, 25), (35, 27), (22, 19), (34, 29)]

def demonstrate_suppression():

    print("Attention Head Suppression Demonstration")
    print("=" * 50)

    top_heads = load_top_heads()
    print(f"Top legal heads: {top_heads}")

    strengths = [0.0, 0.2, 0.5, 0.8, 1.0]
    strategies = ["multiply", "zero", "scale"]

    print(f"\nSuppression strengths to test: {strengths}")
    print(f"Suppression strategies available: {strategies}")

    print("\nReady for integration with model experiments!")

if __name__ == "__main__":
    demonstrate_suppression()
