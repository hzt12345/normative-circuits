#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention Pattern Analysis - 对标 Zhang et al. 2024 Figure 5

分析关键注意力头的注意力模式，理解每个头"看哪里"：
1. 可视化关键头的 attention pattern
2. 分析头对不同 token 类型的注意力分布
3. 区分 "法律概念头" vs "推理逻辑头" vs "答案选项头"

用法:
    python attention_pattern_analysis.py --dataset chinese_legal
    python attention_pattern_analysis.py --dataset chinese_legal --top-k 10
    python attention_pattern_analysis.py --all-datasets
"""

import argparse
import json
import os
import sys
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
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
DEFAULT_TOP_K = 10
DEFAULT_MAX_SAMPLES = 50

DATASET_CONFIG = {
    "chinese_legal": {
        "results_pattern": "output/qwen3-8b/chinese_legal_atp_results_*.json",
        "name": "中文法律",
        "language": "zh",
        "token_patterns": {
            "answer_options": r"[ABCD]",
            "legal_terms": r"(法律|条例|规定|权利|义务|合同|诉讼|裁定|判决|犯罪|刑罚|民事|行政|宪法|法规|法院|检察|公安)",
            "punctuation": r"[，。？！：；、""''（）【】]",
            "numbers": r"\d+",
        }
    },
    "english_legal": {
        "results_pattern": "output/qwen3-8b/english_legal_atp_results_*.json",
        "name": "英文法律",
        "language": "en",
        "token_patterns": {
            "answer_options": r"[ABCD]",
            "legal_terms": r"(law|legal|court|judge|trial|contract|liability|statute|regulation|criminal|civil|plaintiff|defendant|attorney|jurisdiction)",
            "punctuation": r"[.,?!:;\"'()]",
            "numbers": r"\d+",
        }
    },
    "chinese_moral": {
        "results_pattern": "output/qwen3-8b/chinese_moral_atp_results_*.json",
        "name": "中文道德",
        "language": "zh",
        "token_patterns": {
            "answer_options": r"[ABCD]",
            "moral_terms": r"(道德|伦理|正确|错误|善|恶|应该|不应该|责任|义务|权利|公平|正义)",
            "punctuation": r"[，。？！：；、""''（）【】]",
            "numbers": r"\d+",
        }
    },
    "english_moral": {
        "results_pattern": "output/qwen3-8b/english_moral_atp_results_*.json",
        "name": "英文道德",
        "language": "en",
        "token_patterns": {
            "answer_options": r"[ABCD]",
            "moral_terms": r"(moral|ethical|right|wrong|good|bad|should|ought|duty|obligation|fair|just|virtue)",
            "punctuation": r"[.,?!:;\"'()]",
            "numbers": r"\d+",
        }
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


def extract_top_heads(results: Dict, top_k: int = 10) -> List[Tuple[int, int, float]]:
    """从 ATP 结果中提取 top-K 重要头"""
    samples = results.get("results", [])
    if not samples:
        raise ValueError("ATP 结果中没有样本数据")

    first_result = samples[0].get("patching_result")
    if not first_result:
        raise ValueError("ATP 结果格式错误")

    n_layers = len(first_result)
    first_layer_key = list(first_result.keys())[0]
    n_heads = len(first_result[first_layer_key])

    # 构建重要性矩阵
    importance_matrix = defaultdict(list)

    for sample in samples:
        patching_result = sample.get("patching_result")
        if not patching_result:
            continue
        for layer_idx in range(n_layers):
            layer_key = f"layer_{layer_idx}"
            if layer_key not in patching_result:
                continue
            for head_idx in range(n_heads):
                head_key = f"head_{head_idx}"
                if head_key not in patching_result[layer_key]:
                    continue
                prob_diff = patching_result[layer_key][head_key]["prob_diff"]
                if isinstance(prob_diff, list):
                    prob_diff = np.mean(prob_diff)
                importance_matrix[(layer_idx, head_idx)].append(abs(float(prob_diff)))

    # 计算平均重要性
    head_scores = []
    for (layer_idx, head_idx), values in importance_matrix.items():
        score = np.mean(values)
        head_scores.append((layer_idx, head_idx, score))

    head_scores.sort(key=lambda x: x[2], reverse=True)
    return head_scores[:top_k]


def classify_token(token: str, patterns: Dict[str, str]) -> str:
    """根据模式分类 token"""
    for category, pattern in patterns.items():
        if re.search(pattern, token, re.IGNORECASE):
            return category
    return "other"


# =============================================================================
# 注意力模式分析
# =============================================================================

class AttentionPatternAnalyzer:
    """分析注意力头的注意力模式"""

    def __init__(self, model: HookedTransformer, device: str = "cuda"):
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer

    def get_attention_patterns(
        self,
        prompt: str,
        target_heads: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        获取指定头的注意力模式

        Returns:
            Dict mapping (layer, head) to attention pattern tensor [seq_len, seq_len]
        """
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)

        patterns = {}
        for layer_idx, head_idx in target_heads:
            # attention pattern shape: [batch, head, seq_len, seq_len]
            attn_pattern = cache[f"blocks.{layer_idx}.attn.hook_pattern"]
            # 取第一个batch，指定head
            patterns[(layer_idx, head_idx)] = attn_pattern[0, head_idx].cpu()

        return patterns

    def analyze_attention_distribution(
        self,
        prompt: str,
        attention_pattern: torch.Tensor,
        token_patterns: Dict[str, str]
    ) -> Dict[str, float]:
        """
        分析注意力在不同 token 类型上的分布

        Returns:
            Dict mapping token category to average attention weight
        """
        tokens = self.tokenizer.encode(prompt)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # 分类每个 token
        token_categories = [classify_token(t, token_patterns) for t in token_strs]

        # 计算最后一个 token 对每个类别的平均注意力
        last_token_attention = attention_pattern[-1].numpy()  # [seq_len]

        category_attention = defaultdict(list)
        for idx, (attn, cat) in enumerate(zip(last_token_attention, token_categories)):
            category_attention[cat].append(attn)

        # 计算每个类别的平均注意力
        result = {}
        for cat, attns in category_attention.items():
            result[cat] = float(np.mean(attns)) if attns else 0.0

        return result

    def analyze_positional_attention(
        self,
        attention_pattern: torch.Tensor
    ) -> Dict[str, float]:
        """
        分析注意力的位置分布特征

        Returns:
            位置特征字典
        """
        seq_len = attention_pattern.shape[0]
        last_token_attention = attention_pattern[-1].numpy()

        # 计算位置特征
        positions = np.arange(seq_len)

        # 加权平均位置
        weighted_pos = np.sum(positions * last_token_attention) / (np.sum(last_token_attention) + 1e-10)

        # 注意力熵 (衡量注意力的集中程度)
        attn_normalized = last_token_attention / (np.sum(last_token_attention) + 1e-10)
        entropy = -np.sum(attn_normalized * np.log(attn_normalized + 1e-10))

        # 注意力在序列前/中/后三分之一的分布
        third = seq_len // 3
        early_attn = np.sum(last_token_attention[:third])
        middle_attn = np.sum(last_token_attention[third:2*third])
        late_attn = np.sum(last_token_attention[2*third:])

        total_attn = early_attn + middle_attn + late_attn + 1e-10

        return {
            "weighted_position": float(weighted_pos) / seq_len,  # 归一化到 [0, 1]
            "attention_entropy": float(entropy),
            "early_attention_ratio": float(early_attn / total_attn),
            "middle_attention_ratio": float(middle_attn / total_attn),
            "late_attention_ratio": float(late_attn / total_attn),
            "max_attention": float(np.max(last_token_attention)),
            "attention_sparsity": float(np.sum(last_token_attention > 0.1) / seq_len),
        }


def classify_head_function(
    head_stats: Dict[str, Any],
    language: str = "zh"
) -> str:
    """
    根据注意力模式统计分类头的功能

    分类:
    - "answer_head": 主要关注答案选项 (A/B/C/D)
    - "concept_head": 主要关注领域术语 (法律/道德概念)
    - "context_head": 关注上下文信息
    - "position_head": 基于位置的注意力 (如关注最近 token)
    - "mixed_head": 混合功能
    """
    token_attn = head_stats.get("token_attention", {})
    pos_attn = head_stats.get("positional_attention", {})

    # 获取对答案选项的注意力
    answer_attn = token_attn.get("answer_options", 0)

    # 获取对领域术语的注意力
    if language == "zh":
        term_attn = token_attn.get("legal_terms", 0) + token_attn.get("moral_terms", 0)
    else:
        term_attn = token_attn.get("legal_terms", 0) + token_attn.get("moral_terms", 0)

    # 获取位置特征
    late_ratio = pos_attn.get("late_attention_ratio", 0)
    entropy = pos_attn.get("attention_entropy", 0)

    # 分类逻辑
    if answer_attn > 0.15:
        return "answer_head"
    elif term_attn > 0.2:
        return "concept_head"
    elif late_ratio > 0.6:
        return "position_head"
    elif entropy < 2.0:  # 低熵表示注意力集中
        return "focused_head"
    else:
        return "context_head"


# =============================================================================
# 主分析流程
# =============================================================================

def run_attention_pattern_analysis(
    dataset_key: str,
    top_k: int = 10,
    max_samples: int = 50,
    output_dir: str = None,
    model_cache: str = "Model",
    device: str = "cuda"
) -> Dict[str, Any]:
    """运行注意力模式分析"""

    if output_dir is None:
        output_dir = PROJECT_ROOT / "output" / "qwen3-8b" / "attention_patterns"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIG[dataset_key]

    print("=" * 70)
    print(f"Attention Pattern Analysis - {config['name']}")
    print("=" * 70)

    set_all_seeds(RANDOM_SEED)

    # 1. 加载 ATP 结果
    print("\n[1/4] 加载 ATP 分析结果...")
    atp_results = load_atp_results(dataset_key)

    # 2. 提取 top-K 头
    print("\n[2/4] 提取重要头...")
    top_heads = extract_top_heads(atp_results, top_k)

    print(f"\nTop-{top_k} 重要头:")
    for i, (layer, head, score) in enumerate(top_heads):
        print(f"  {i+1}. Layer {layer}, Head {head}: {score:.6f}")

    # 3. 加载模型
    print("\n[3/4] 加载模型...")
    if device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        device = "cpu"

    model = load_hooked_transformer("Qwen/Qwen3-8B", device=device)

    analyzer = AttentionPatternAnalyzer(model, device)

    # 4. 分析注意力模式
    print("\n[4/4] 分析注意力模式...")
    samples = atp_results.get("results", [])

    if max_samples and max_samples < len(samples):
        import random
        sample_indices = random.sample(range(len(samples)), max_samples)
        samples = [samples[i] for i in sample_indices]

    # 收集每个头的统计数据
    head_statistics = {(l, h): {
        "token_attention": defaultdict(list),
        "positional_attention": defaultdict(list),
    } for l, h, _ in top_heads}

    target_heads = [(l, h) for l, h, _ in top_heads]

    for sample in tqdm(samples, desc="Analyzing samples"):
        prompt = sample["sample"]["correct_prompt"]

        try:
            patterns = analyzer.get_attention_patterns(prompt, target_heads)

            for (layer, head), pattern in patterns.items():
                # Token 类型注意力分布
                token_attn = analyzer.analyze_attention_distribution(
                    prompt, pattern, config["token_patterns"]
                )
                for cat, attn in token_attn.items():
                    head_statistics[(layer, head)]["token_attention"][cat].append(attn)

                # 位置注意力分布
                pos_attn = analyzer.analyze_positional_attention(pattern)
                for feat, val in pos_attn.items():
                    head_statistics[(layer, head)]["positional_attention"][feat].append(val)

        except Exception as e:
            print(f"警告: 分析样本时出错: {e}")
            continue

    # 计算平均统计
    results = {
        "metadata": {
            "dataset": dataset_key,
            "dataset_name": config["name"],
            "language": config["language"],
            "top_k": top_k,
            "n_samples_analyzed": len(samples),
            "timestamp": datetime.now().isoformat(),
        },
        "head_analysis": []
    }

    for layer, head, importance_score in top_heads:
        stats = head_statistics[(layer, head)]

        # 计算平均值
        avg_token_attn = {
            cat: float(np.mean(vals)) if vals else 0.0
            for cat, vals in stats["token_attention"].items()
        }

        avg_pos_attn = {
            feat: float(np.mean(vals)) if vals else 0.0
            for feat, vals in stats["positional_attention"].items()
        }

        # 分类头的功能
        head_stats = {
            "token_attention": avg_token_attn,
            "positional_attention": avg_pos_attn,
        }
        function_type = classify_head_function(head_stats, config["language"])

        results["head_analysis"].append({
            "layer": layer,
            "head": head,
            "importance_score": importance_score,
            "function_type": function_type,
            "token_attention_distribution": avg_token_attn,
            "positional_features": avg_pos_attn,
        })

    # 统计功能分布
    function_counts = defaultdict(int)
    for head_info in results["head_analysis"]:
        function_counts[head_info["function_type"]] += 1

    results["summary"] = {
        "function_distribution": dict(function_counts),
        "total_heads_analyzed": len(top_heads),
    }

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{dataset_key}_attention_patterns_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {json_path}")

    # 生成报告
    report = generate_report(results)
    report_path = output_dir / f"{dataset_key}_attention_patterns_report_{timestamp}.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"报告已保存: {report_path}")
    print("\n" + report)

    # 生成可视化
    try:
        generate_visualization(results, output_dir, dataset_key, timestamp)
    except Exception as e:
        print(f"警告: 生成可视化失败: {e}")

    return results


def generate_report(results: Dict[str, Any]) -> str:
    """生成分析报告"""
    lines = [
        "=" * 70,
        f"Attention Pattern Analysis Report - {results['metadata']['dataset_name']}",
        "=" * 70,
        f"生成时间: {results['metadata']['timestamp']}",
        f"分析样本数: {results['metadata']['n_samples_analyzed']}",
        f"分析头数: {results['metadata']['top_k']}",
        "",
        "=" * 70,
        "功能分类统计",
        "=" * 70,
    ]

    for func_type, count in results["summary"]["function_distribution"].items():
        lines.append(f"  {func_type}: {count} 个头")

    lines.extend([
        "",
        "=" * 70,
        "各头详细分析",
        "=" * 70,
    ])

    for head_info in results["head_analysis"]:
        lines.extend([
            "",
            f"Layer {head_info['layer']}, Head {head_info['head']}",
            "-" * 40,
            f"  重要性分数: {head_info['importance_score']:.6f}",
            f"  功能分类: {head_info['function_type']}",
            "",
            "  Token 类型注意力分布:",
        ])

        for cat, attn in sorted(head_info["token_attention_distribution"].items(),
                                key=lambda x: x[1], reverse=True):
            lines.append(f"    {cat}: {attn:.4f}")

        lines.append("")
        lines.append("  位置特征:")

        pos_feats = head_info["positional_features"]
        lines.extend([
            f"    加权位置: {pos_feats.get('weighted_position', 0):.3f}",
            f"    注意力熵: {pos_feats.get('attention_entropy', 0):.3f}",
            f"    前段注意力: {pos_feats.get('early_attention_ratio', 0):.3f}",
            f"    中段注意力: {pos_feats.get('middle_attention_ratio', 0):.3f}",
            f"    后段注意力: {pos_feats.get('late_attention_ratio', 0):.3f}",
        ])

    lines.extend([
        "",
        "=" * 70,
        "功能解释",
        "=" * 70,
        "  answer_head: 主要关注答案选项 (A/B/C/D)",
        "  concept_head: 主要关注领域术语 (法律/道德概念)",
        "  context_head: 关注上下文信息",
        "  position_head: 基于位置的注意力 (如关注最近 token)",
        "  focused_head: 注意力集中在特定位置",
        "=" * 70,
    ])

    return "\n".join(lines)


def generate_visualization(
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

    # 提取数据
    heads = results["head_analysis"]

    # 1. 功能分布饼图
    func_dist = results["summary"]["function_distribution"]

    fig1 = go.Figure(data=[go.Pie(
        labels=list(func_dist.keys()),
        values=list(func_dist.values()),
        hole=0.3
    )])
    fig1.update_layout(
        title=f"Head Function Distribution - {results['metadata']['dataset_name']}",
        width=600,
        height=400
    )
    fig1.write_html(output_dir / f"{dataset_key}_function_distribution_{timestamp}.html")

    # 2. Token 注意力热图
    categories = set()
    for h in heads:
        categories.update(h["token_attention_distribution"].keys())
    categories = sorted(categories)

    head_labels = [f"L{h['layer']}H{h['head']}" for h in heads]

    heatmap_data = []
    for h in heads:
        row = [h["token_attention_distribution"].get(cat, 0) for cat in categories]
        heatmap_data.append(row)

    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=categories,
        y=head_labels,
        colorscale='Blues'
    ))
    fig2.update_layout(
        title=f"Token Type Attention Distribution - {results['metadata']['dataset_name']}",
        xaxis_title="Token Category",
        yaxis_title="Attention Head",
        width=900,
        height=600
    )
    fig2.write_html(output_dir / f"{dataset_key}_token_attention_heatmap_{timestamp}.html")

    # 3. 位置注意力特征条形图
    fig3 = make_subplots(rows=1, cols=3, subplot_titles=[
        "Early Attention", "Middle Attention", "Late Attention"
    ])

    early_vals = [h["positional_features"].get("early_attention_ratio", 0) for h in heads]
    middle_vals = [h["positional_features"].get("middle_attention_ratio", 0) for h in heads]
    late_vals = [h["positional_features"].get("late_attention_ratio", 0) for h in heads]

    fig3.add_trace(go.Bar(x=head_labels, y=early_vals, name="Early"), row=1, col=1)
    fig3.add_trace(go.Bar(x=head_labels, y=middle_vals, name="Middle"), row=1, col=2)
    fig3.add_trace(go.Bar(x=head_labels, y=late_vals, name="Late"), row=1, col=3)

    fig3.update_layout(
        title=f"Positional Attention Distribution - {results['metadata']['dataset_name']}",
        showlegend=False,
        width=1200,
        height=400
    )
    fig3.write_html(output_dir / f"{dataset_key}_positional_attention_{timestamp}.html")

    print(f"可视化图表已保存到: {output_dir}")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Attention Pattern Analysis - Zhang et al. 2024 Figure 5 Style",
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
        "--all-datasets",
        action="store_true",
        help="分析所有数据集"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"分析 top-K 重要头 (默认: {DEFAULT_TOP_K})"
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

    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_datasets:
        datasets = list(DATASET_CONFIG.keys())
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        try:
            run_attention_pattern_analysis(
                dataset_key=dataset,
                top_k=args.top_k,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
                model_cache=args.model_cache,
                device=args.device
            )
        except Exception as e:
            print(f"\n错误: 处理 {dataset} 时发生异常: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
