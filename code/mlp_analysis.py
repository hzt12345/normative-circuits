#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP Layer Analysis - 对标 Zhang et al. 2024 Figure 6

分析 MLP 层在推理中的作用：
1. MLP 层的 Path Patching (重要性分析)
2. MLP 输入/输出与关键 token 的相似度分析
3. 理解 MLP 如何逐层处理信息

用法:
    python mlp_analysis.py --dataset chinese_legal --method patching
    python mlp_analysis.py --dataset chinese_legal --method similarity
    python mlp_analysis.py --dataset chinese_legal --method all
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from tqdm import tqdm

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from reproducibility import set_all_seeds
from local_model_loader import load_hooked_transformer

# =============================================================================
# 配置
# =============================================================================

RANDOM_SEED = 42
DEFAULT_MAX_SAMPLES = 100

DATASET_CONFIG = {
    "chinese_legal": {
        "results_pattern": "output/qwen3-8b/chinese_legal_atp_results_*.json",
        "name": "中文法律",
    },
    "english_legal": {
        "results_pattern": "output/qwen3-8b/english_legal_atp_results_*.json",
        "name": "英文法律",
    },
    "chinese_moral": {
        "results_pattern": "output/qwen3-8b/chinese_moral_atp_results_*.json",
        "name": "中文道德",
    },
    "english_moral": {
        "results_pattern": "output/qwen3-8b/english_moral_atp_results_*.json",
        "name": "英文道德",
    },
    "general_mcqa": {
        "results_pattern": "output/qwen3-8b/general_mcqa_atp_results_*.json",
        "name": "通用MCQA(STEM)",
    },
}


# =============================================================================
# 工具函数
# =============================================================================

def find_latest_file(pattern: str) -> Optional[Path]:
    """找到匹配模式的最新文件"""
    import glob
    matches = sorted(glob.glob(str(PROJECT_ROOT / pattern)), key=os.path.getmtime, reverse=True)
    return Path(matches[0]) if matches else None


def load_atp_results(dataset_key: str) -> Dict:
    """加载 ATP 分析结果"""
    config = DATASET_CONFIG[dataset_key]
    results_file = find_latest_file(config["results_pattern"])
    if not results_file:
        raise FileNotFoundError(f"未找到 {dataset_key} 的 ATP 结果文件")
    print(f"加载 ATP 结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """计算余弦相似度"""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    if torch.norm(a_flat) < 1e-8 or torch.norm(b_flat) < 1e-8:
        return 0.0
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


# =============================================================================
# MLP Path Patching
# =============================================================================

class MLPPathPatcher:
    """MLP 层 Path Patching 分析"""

    def __init__(self, model: HookedTransformer, device: str = "cuda"):
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer
        self.n_layers = model.cfg.n_layers

    def run_mlp_patching(
        self,
        correct_prompt: str,
        corrupted_prompt: str,
        answer: str,
        position: str = "all",
    ) -> Dict[str, float]:
        """
        对每个 MLP 层进行 Path Patching

        Args:
            position: patching 位置
                - "all": 所有位置 (默认, 原始行为)
                - "end": 仅最后一个 token
                - "answer": 答案选项区域 (最后 ~20% tokens)

        Returns:
            Dict mapping layer index to probability difference
        """
        # Tokenize
        correct_tokens = self.tokenizer.encode(correct_prompt, return_tensors="pt").to(self.device)
        corrupted_tokens = self.tokenizer.encode(corrupted_prompt, return_tensors="pt").to(self.device)

        # Get answer token
        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        if not answer_tokens:
            return {}
        answer_token = answer_tokens[0]

        # Get clean and corrupted caches
        with torch.no_grad():
            _, clean_cache = self.model.run_with_cache(correct_tokens)
            clean_logits = self.model(correct_tokens)
            clean_prob = torch.softmax(clean_logits[0, -1], dim=-1)[answer_token].item()

            _, corrupted_cache = self.model.run_with_cache(corrupted_tokens)

        # Determine position range for patching
        seq_len = correct_tokens.shape[1]
        if position == "end":
            patch_start = seq_len - 1
            patch_end = seq_len
        elif position == "answer":
            # Answer region: last 20% of tokens (where options typically are)
            patch_start = max(0, int(seq_len * 0.8))
            patch_end = seq_len
        else:  # "all"
            patch_start = 0
            patch_end = seq_len

        results = {}

        # Patch each MLP layer
        for layer_idx in range(self.n_layers):
            hook_name = f"blocks.{layer_idx}.hook_mlp_out"

            # Get the clean MLP output
            clean_mlp_out = clean_cache[hook_name]

            def patch_hook(activation, hook, clean_act=clean_mlp_out,
                          p_start=patch_start, p_end=patch_end):
                # Replace corrupted activation with clean at specified positions
                min_len = min(activation.shape[1], clean_act.shape[1])
                actual_start = min(p_start, min_len)
                actual_end = min(p_end, min_len)
                activation[:, actual_start:actual_end, :] = clean_act[:, actual_start:actual_end, :]
                return activation

            # Run with patched MLP
            with torch.no_grad():
                patched_logits = self.model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, patch_hook)]
                )
                patched_prob = torch.softmax(patched_logits[0, -1], dim=-1)[answer_token].item()

            # Calculate effect
            results[layer_idx] = patched_prob - clean_prob

        return results


# =============================================================================
# MLP Similarity Analysis
# =============================================================================

class MLPSimilarityAnalyzer:
    """MLP 输入/输出相似度分析"""

    def __init__(self, model: HookedTransformer, device: str = "cuda"):
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer
        self.n_layers = model.cfg.n_layers

    def get_token_embedding(self, token_str: str) -> torch.Tensor:
        """获取 token 的嵌入向量"""
        token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
        if not token_ids:
            return None
        # Use W_U (unembedding) since MLP outputs are in residual stream space
        embedding = self.model.W_U[:, token_ids[0]]
        return embedding

    def analyze_mlp_similarity(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        分析 MLP 各层输入/输出与答案 token 的相似度

        Returns:
            各层相似度数据
        """
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Get answer embedding
        answer_embedding = self.get_token_embedding(answer)
        if answer_embedding is None:
            return {}

        # Get cache with all MLP activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)

        results = {
            "mlp_in_similarity": [],
            "mlp_out_similarity": [],
            "residual_similarity": [],
        }

        for layer_idx in range(self.n_layers):
            # Residual stream before MLP (in residual stream space, d_model=4096)
            mlp_in = cache[f"blocks.{layer_idx}.hook_resid_mid"]
            # MLP output (in residual stream space, d_model=4096)
            mlp_out = cache[f"blocks.{layer_idx}.hook_mlp_out"]
            # Residual stream after full layer
            resid_mid = cache[f"blocks.{layer_idx}.hook_resid_post"]

            # Calculate similarity at last token position
            mlp_in_last = mlp_in[0, -1]
            mlp_out_last = mlp_out[0, -1]
            resid_mid_last = resid_mid[0, -1]

            # Cosine similarity with answer embedding
            sim_in = cosine_similarity(mlp_in_last.cpu(), answer_embedding.cpu())
            sim_out = cosine_similarity(mlp_out_last.cpu(), answer_embedding.cpu())
            sim_resid = cosine_similarity(resid_mid_last.cpu(), answer_embedding.cpu())

            results["mlp_in_similarity"].append(sim_in)
            results["mlp_out_similarity"].append(sim_out)
            results["residual_similarity"].append(sim_resid)

        return results


# =============================================================================
# 主分析流程
# =============================================================================

def run_mlp_analysis(
    dataset_key: str,
    method: str = "all",
    max_samples: int = 100,
    output_dir: str = None,
    model_cache: str = "Model",
    device: str = "cuda",
    position: str = "all",
) -> Dict[str, Any]:
    """运行 MLP 分析"""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "qwen3-8b" / "mlp_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIG[dataset_key]

    print("=" * 70)
    print(f"MLP Analysis - {config['name']}")
    print(f"Method: {method}, Position: {position}")
    print("=" * 70)

    set_all_seeds(RANDOM_SEED)

    # 1. 加载 ATP 结果
    print("\n[1/3] 加载 ATP 分析结果...")
    atp_results = load_atp_results(dataset_key)
    samples = atp_results.get("results", [])

    if max_samples and max_samples < len(samples):
        import random
        sample_indices = random.sample(range(len(samples)), max_samples)
        samples = [samples[i] for i in sample_indices]

    print(f"分析样本数: {len(samples)}")

    # 2. 加载模型
    print("\n[2/3] 加载模型...")
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        device = "cpu"

    model = load_hooked_transformer("Qwen/Qwen3-8B", device=device)

    n_layers = model.cfg.n_layers
    print(f"模型层数: {n_layers}")

    results = {
        "metadata": {
            "dataset": dataset_key,
            "dataset_name": config["name"],
            "method": method,
            "position": position,
            "n_samples": len(samples),
            "n_layers": n_layers,
            "timestamp": datetime.now().isoformat(),
        }
    }

    # 3. 运行分析
    print("\n[3/3] 运行 MLP 分析...")

    if method in ["patching", "all"]:
        print("\n--- MLP Path Patching ---")
        patcher = MLPPathPatcher(model, device)

        patching_results = defaultdict(list)

        for sample in tqdm(samples, desc="MLP Patching"):
            try:
                correct_prompt = sample["sample"]["correct_prompt"]
                corrupted_prompt = sample["sample"]["incorrect_prompt"]
                answer = sample["sample"]["answer"]

                layer_effects = patcher.run_mlp_patching(correct_prompt, corrupted_prompt, answer, position=position)

                for layer_idx, effect in layer_effects.items():
                    patching_results[layer_idx].append(effect)

            except Exception as e:
                continue

        # 计算平均效应
        results["mlp_patching"] = {
            "layer_effects": {
                str(layer): {
                    "mean": float(np.mean(effects)),
                    "std": float(np.std(effects)),
                    "abs_mean": float(np.mean(np.abs(effects))),
                }
                for layer, effects in sorted(patching_results.items())
            }
        }

        # 找到最重要的 MLP 层
        layer_importance = [(l, np.mean(np.abs(e))) for l, e in patching_results.items()]
        layer_importance.sort(key=lambda x: x[1], reverse=True)

        results["mlp_patching"]["top_layers"] = [
            {"layer": l, "importance": float(imp)}
            for l, imp in layer_importance[:10]
        ]

    if method in ["similarity", "all"]:
        print("\n--- MLP Similarity Analysis ---")
        analyzer = MLPSimilarityAnalyzer(model, device)

        similarity_results = {
            "mlp_in": defaultdict(list),
            "mlp_out": defaultdict(list),
            "residual": defaultdict(list),
        }

        for sample in tqdm(samples, desc="MLP Similarity"):
            try:
                correct_prompt = sample["sample"]["correct_prompt"]
                answer = sample["sample"]["answer"]

                sim_data = analyzer.analyze_mlp_similarity(correct_prompt, answer)

                if sim_data:
                    for layer_idx in range(n_layers):
                        if layer_idx < len(sim_data.get("mlp_in_similarity", [])):
                            similarity_results["mlp_in"][layer_idx].append(
                                sim_data["mlp_in_similarity"][layer_idx]
                            )
                            similarity_results["mlp_out"][layer_idx].append(
                                sim_data["mlp_out_similarity"][layer_idx]
                            )
                            similarity_results["residual"][layer_idx].append(
                                sim_data["residual_similarity"][layer_idx]
                            )

            except Exception as e:
                print(f"  Warning (similarity): {e}")
                continue

        # 计算平均相似度
        results["mlp_similarity"] = {
            "mlp_in_by_layer": [
                float(np.mean(similarity_results["mlp_in"][l]))
                for l in range(n_layers)
            ],
            "mlp_out_by_layer": [
                float(np.mean(similarity_results["mlp_out"][l]))
                for l in range(n_layers)
            ],
            "residual_by_layer": [
                float(np.mean(similarity_results["residual"][l]))
                for l in range(n_layers)
            ],
        }

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{dataset_key}_mlp_analysis_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {json_path}")

    # 生成报告
    report = generate_mlp_report(results)
    report_path = output_dir / f"{dataset_key}_mlp_report_{timestamp}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存: {report_path}")
    print("\n" + report)

    # 生成可视化
    try:
        generate_mlp_visualization(results, output_dir, dataset_key, timestamp)
    except Exception as e:
        print(f"警告: 生成可视化失败: {e}")

    return results


def generate_mlp_report(results: Dict[str, Any]) -> str:
    """生成 MLP 分析报告"""
    lines = [
        "=" * 70,
        f"MLP Analysis Report - {results['metadata']['dataset_name']}",
        "=" * 70,
        f"生成时间: {results['metadata']['timestamp']}",
        f"分析方法: {results['metadata']['method']}",
        f"分析样本数: {results['metadata']['n_samples']}",
        f"模型层数: {results['metadata']['n_layers']}",
        "",
    ]

    if "mlp_patching" in results:
        lines.extend([
            "=" * 70,
            "MLP Path Patching 结果",
            "=" * 70,
            "",
            "Top-10 最重要的 MLP 层:",
        ])

        for item in results["mlp_patching"]["top_layers"]:
            lines.append(f"  Layer {item['layer']}: {item['importance']:.6f}")

        lines.extend([
            "",
            "各层效应 (按层顺序):",
        ])

        for layer, data in sorted(results["mlp_patching"]["layer_effects"].items(), key=lambda x: int(x[0])):
            lines.append(f"  Layer {layer}: mean={data['mean']:.6f}, abs_mean={data['abs_mean']:.6f}")

    if "mlp_similarity" in results:
        lines.extend([
            "",
            "=" * 70,
            "MLP 相似度分析结果",
            "=" * 70,
            "",
            "各层 MLP 输出与答案的相似度:",
        ])

        mlp_out_sim = results["mlp_similarity"]["mlp_out_by_layer"]

        # 找到相似度最高的层
        max_sim_layer = np.argmax(mlp_out_sim)
        lines.append(f"  相似度最高层: Layer {max_sim_layer} ({mlp_out_sim[max_sim_layer]:.4f})")

        # 显示每层相似度
        lines.append("")
        lines.append("  层级相似度变化 (MLP Output):")
        for i, sim in enumerate(mlp_out_sim):
            bar = "█" * int(sim * 50) if sim > 0 else ""
            lines.append(f"    Layer {i:2d}: {sim:+.4f} {bar}")

    lines.extend([
        "",
        "=" * 70,
        "结论",
        "=" * 70,
    ])

    if "mlp_patching" in results:
        top_layers = [item['layer'] for item in results["mlp_patching"]["top_layers"][:3]]
        lines.append(f"  最关键 MLP 层: {top_layers}")

    if "mlp_similarity" in results:
        mlp_out_sim = results["mlp_similarity"]["mlp_out_by_layer"]
        # 找到相似度开始显著增加的层
        for i in range(len(mlp_out_sim) - 1):
            if mlp_out_sim[i+1] - mlp_out_sim[i] > 0.02:
                lines.append(f"  答案信息开始积累: Layer {i}")
                break

    lines.append("=" * 70)

    return "\n".join(lines)


def generate_mlp_visualization(
    results: Dict[str, Any],
    output_dir: Path,
    dataset_key: str,
    timestamp: str
):
    """生成可视化图表"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("警告: plotly 未安装，跳过可视化")
        return

    n_layers = results["metadata"]["n_layers"]
    layers = list(range(n_layers))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["MLP Layer Importance (Path Patching)", "MLP Output Similarity to Answer"],
        vertical_spacing=0.15
    )

    if "mlp_patching" in results:
        importance = [
            results["mlp_patching"]["layer_effects"].get(str(l), {}).get("abs_mean", 0)
            for l in layers
        ]
        fig.add_trace(
            go.Bar(x=layers, y=importance, name="MLP Importance", marker_color='steelblue'),
            row=1, col=1
        )

    if "mlp_similarity" in results:
        mlp_out_sim = results["mlp_similarity"]["mlp_out_by_layer"]
        fig.add_trace(
            go.Scatter(x=layers, y=mlp_out_sim, mode='lines+markers', name="MLP Out Similarity",
                      line=dict(color='coral', width=2)),
            row=2, col=1
        )

    fig.update_layout(
        title=f"MLP Layer Analysis - {results['metadata']['dataset_name']}",
        height=700,
        width=1000,
        showlegend=True
    )

    fig.update_xaxes(title_text="Layer", row=2, col=1)
    fig.update_yaxes(title_text="Importance (abs mean)", row=1, col=1)
    fig.update_yaxes(title_text="Cosine Similarity", row=2, col=1)

    fig.write_html(output_dir / f"{dataset_key}_mlp_analysis_{timestamp}.html")
    print(f"可视化已保存: {output_dir / f'{dataset_key}_mlp_analysis_{timestamp}.html'}")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MLP Layer Analysis - Zhang et al. 2024 Figure 6 Style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIG.keys()),
        default="chinese_legal",
        help="要分析的数据集"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["patching", "similarity", "all"],
        default="all",
        help="分析方法"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"最大分析样本数 (默认: {DEFAULT_MAX_SAMPLES})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录"
    )

    parser.add_argument(
        "--model-cache",
        type=str,
        default="Model",
        help="模型缓存目录"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备"
    )

    parser.add_argument(
        "--position",
        type=str,
        choices=["all", "end", "answer"],
        default="all",
        help="A6: MLP patching 位置 - 'all' (全序列), 'end' (最后token), 'answer' (答案区域)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        run_mlp_analysis(
            dataset_key=args.dataset,
            method=args.method,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
            model_cache=args.model_cache,
            device=args.device,
            position=args.position,
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
