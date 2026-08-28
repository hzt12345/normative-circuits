

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AttentionHeadAnalyzer:

    def __init__(self, results_dir: str = "output/01-China-Law"):
        self.results_dir = results_dir
        self.results = None
        self.importance_matrix = None
        self.head_rankings = None

    def load_results(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if file_path is None:

            file_options = [
                os.path.join(self.results_dir, "results-latest.pkl"),
                os.path.join(self.results_dir, "results-20250707_135018.pkl"),
                os.path.join(self.results_dir, "results-incremental-20250707_075941.pkl"),
                os.path.join(self.results_dir, "results-recovered.json")
            ]

            for file_path in file_options:
                if os.path.exists(file_path):
                    logger.info(f"Found results file: {file_path}")
                    break
            else:
                raise FileNotFoundError(f"No results files found in {self.results_dir}")

        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Loaded {len(results)} samples from pickle file: {file_path}")
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Loaded {len(results)} samples from JSON file: {file_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            self.results = results
            return results

        except Exception as e:
            logger.error(f"Failed to load results from {file_path}: {e}")
            raise

    def extract_importance_matrix(self) -> np.ndarray:
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")

        first_result = self.results[0]["patching_result"]
        n_layers = len(first_result)
        first_layer_key = list(first_result.keys())[0]
        n_heads = len(first_result[first_layer_key])
        n_samples = len(self.results)

        logger.info(f"Processing {n_samples} samples with {n_layers} layers and {n_heads} heads each")

        importance_matrix = np.zeros((n_samples, n_layers, n_heads))

        for sample_idx, result in enumerate(self.results):
            patching_result = result["patching_result"]

            for layer_idx in range(n_layers):
                layer_key = f"layer_{layer_idx}"
                if layer_key not in patching_result:
                    logger.warning(f"Missing {layer_key} in sample {sample_idx}")
                    continue

                for head_idx in range(n_heads):
                    head_key = f"head_{head_idx}"
                    if head_key not in patching_result[layer_key]:
                        logger.warning(f"Missing {head_key} in {layer_key} of sample {sample_idx}")
                        continue

                    prob_diff = patching_result[layer_key][head_key]["prob_diff"]

                    if isinstance(prob_diff, np.ndarray):
                        prob_diff = prob_diff.item()

                    importance_matrix[sample_idx, layer_idx, head_idx] = prob_diff

        self.importance_matrix = importance_matrix
        logger.info(f"Extracted importance matrix with shape: {importance_matrix.shape}")

        return importance_matrix

    def rank_heads_by_importance(self,
                                method: str = "mean_abs",
                                top_k: int = 5) -> List[Tuple[int, int, float]]:
        if self.importance_matrix is None:
            raise ValueError("Importance matrix not extracted. Call extract_importance_matrix() first.")

        n_samples, n_layers, n_heads = self.importance_matrix.shape
        head_scores = []

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                head_values = self.importance_matrix[:, layer_idx, head_idx]

                if method == "mean_abs":
                    score = np.mean(np.abs(head_values))
                elif method == "max_abs":
                    score = np.max(np.abs(head_values))
                elif method == "mean":
                    score = np.mean(head_values)
                elif method == "std":
                    score = np.std(head_values)
                elif method == "consistency":

                    sample_ranks = []
                    for sample_idx in range(n_samples):
                        sample_importance = np.abs(self.importance_matrix[sample_idx])
                        flat_importance = sample_importance.reshape(-1)
                        head_flat_idx = layer_idx * n_heads + head_idx
                        rank = np.argsort(flat_importance)[::-1]
                        position = np.where(rank == head_flat_idx)[0][0]
                        sample_ranks.append(position)

                    score = np.mean(np.array(sample_ranks) < top_k)
                else:
                    raise ValueError(f"Unknown ranking method: {method}")

                head_scores.append((layer_idx, head_idx, score))

        head_scores.sort(key=lambda x: x[2], reverse=True)

        self.head_rankings = {
            "method": method,
            "top_k": top_k,
            "rankings": head_scores[:top_k],
            "full_rankings": head_scores
        }

        logger.info(f"Ranked heads using method '{method}'. Top {top_k} heads:")
        for i, (layer, head, score) in enumerate(head_scores[:top_k]):
            logger.info(f"  {i+1}. Layer {layer}, Head {head}: {score:.6f}")

        return head_scores[:top_k]

    def get_top_heads(self, method: str = "mean_abs", top_k: int = 5) -> List[Tuple[int, int]]:
        rankings = self.rank_heads_by_importance(method=method, top_k=top_k)
        return [(layer, head) for layer, head, score in rankings]

    def save_analysis(self, output_dir: str = "output/02-Moral-Analysis") -> str:
        os.makedirs(output_dir, exist_ok=True)

        if self.head_rankings is None:
            raise ValueError("No head rankings available. Call rank_heads_by_importance() first.")

        analysis_summary = {
            "analysis_date": datetime.now().isoformat(),
            "n_samples": len(self.results) if self.results else 0,
            "n_layers": self.importance_matrix.shape[1] if self.importance_matrix is not None else 0,
            "n_heads": self.importance_matrix.shape[2] if self.importance_matrix is not None else 0,
            "ranking_method": self.head_rankings["method"],
            "top_k": self.head_rankings["top_k"],
            "top_heads": [
                {
                    "rank": i + 1,
                    "layer": layer,
                    "head": head,
                    "importance_score": score
                }
                for i, (layer, head, score) in enumerate(self.head_rankings["rankings"])
            ],
            "all_head_rankings": [
                {
                    "layer": layer,
                    "head": head,
                    "importance_score": score
                }
                for layer, head, score in self.head_rankings["full_rankings"]
            ]
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = os.path.join(output_dir, f"attention_head_analysis_{timestamp}.json")

        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis saved to: {analysis_file}")

        top_heads_file = os.path.join(output_dir, "top_legal_heads.json")
        top_heads_data = {
            "analysis_date": datetime.now().isoformat(),
            "ranking_method": self.head_rankings["method"],
            "top_heads": [(layer, head) for layer, head, score in self.head_rankings["rankings"]]
        }

        with open(top_heads_file, 'w', encoding='utf-8') as f:
            json.dump(top_heads_data, f, indent=2)

        logger.info(f"Top heads saved to: {top_heads_file}")

        return analysis_file

    def create_importance_heatmap(self, method: str = "mean_abs") -> np.ndarray:
        if self.importance_matrix is None:
            raise ValueError("Importance matrix not extracted. Call extract_importance_matrix() first.")

        n_samples, n_layers, n_heads = self.importance_matrix.shape
        heatmap = np.zeros((n_layers, n_heads))

        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                head_values = self.importance_matrix[:, layer_idx, head_idx]

                if method == "mean_abs":
                    score = np.mean(np.abs(head_values))
                elif method == "max_abs":
                    score = np.max(np.abs(head_values))
                elif method == "mean":
                    score = np.mean(head_values)
                elif method == "std":
                    score = np.std(head_values)
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")

                heatmap[layer_idx, head_idx] = score

        return heatmap

def main():

    analyzer = AttentionHeadAnalyzer()

    print("Loading path patching results...")
    results = analyzer.load_results()

    print("Extracting importance matrix...")
    importance_matrix = analyzer.extract_importance_matrix()

    methods = ["mean_abs", "max_abs", "consistency"]

    for method in methods:
        print(f"\n=== Ranking by {method} ===")
        top_heads = analyzer.rank_heads_by_importance(method=method, top_k=5)

        print(f"Top 5 heads by {method}:")
        for i, (layer, head, score) in enumerate(top_heads):
            print(f"  {i+1}. Layer {layer}, Head {head}: {score:.6f}")

    print("\nSaving analysis results...")
    analysis_file = analyzer.save_analysis()
    print(f"Analysis saved to: {analysis_file}")

    print("\nCreating importance heatmap...")
    heatmap = analyzer.create_importance_heatmap(method="mean_abs")
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Max importance: {np.max(heatmap):.6f}")
    print(f"Min importance: {np.min(heatmap):.6f}")

if __name__ == "__main__":
    main()
