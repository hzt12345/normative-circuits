#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JEC-QA 数据处理脚本

将 HuggingFace 上的 AGIEval JEC-QA 数据集转换为 Path Patching 格式
"""

import os
import re
import json
import random
import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


# 法律关键术语（用于 masking）
LEGAL_TERMS_ZH = [
    '法律', '法规', '条例', '规定', '民法', '刑法', '宪法', '行政法',
    '诉讼', '起诉', '上诉', '判决', '裁定', '执行', '强制', '管辖',
    '原告', '被告', '当事人', '代理人', '律师', '法官', '检察官',
    '权利', '义务', '责任', '赔偿', '违约', '侵权', '犯罪', '处罚',
    '合同', '协议', '合法', '违法', '有效', '无效', '撤销', '解除',
    '公司', '股东', '董事', '监事', '经理', '法人', '股权', '出资',
    '财产', '债权', '债务', '担保', '抵押', '质押', '留置', '保证',
    '继承', '遗嘱', '遗产', '婚姻', '离婚', '抚养', '赡养', '监护',
    '行政', '许可', '处罚', '强制', '复议', '诉讼', '赔偿', '补偿',
    '知识产权', '专利', '商标', '著作权', '版权', '发明', '创作',
]


def create_masked_prompt(text: str, mask_ratio: float = 0.3) -> str:
    """创建 mask 版本的 prompt"""
    masked_text = text

    terms_found = []
    for term in LEGAL_TERMS_ZH:
        for match in re.finditer(re.escape(term), text):
            terms_found.append((match.start(), match.end(), match.group()))

    terms_found.sort(key=lambda x: x[0], reverse=True)

    n_to_mask = max(1, int(len(terms_found) * mask_ratio))
    if terms_found:
        terms_to_mask = random.sample(terms_found, min(n_to_mask, len(terms_found)))

        for start, end, term in terms_to_mask:
            placeholder = f"**{chr(65 + random.randint(0, 25))}**"
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

    return masked_text


def parse_choices(choices_value) -> List[str]:
    """解析选项"""
    # numpy 数组
    if hasattr(choices_value, 'tolist'):
        return [str(c) for c in choices_value.tolist()]

    # 列表
    if isinstance(choices_value, list):
        return [str(c) for c in choices_value]

    # 字符串
    if isinstance(choices_value, str):
        try:
            choices = ast.literal_eval(choices_value)
            if isinstance(choices, list):
                return [str(c) for c in choices]
        except:
            pass

        # 尝试手动解析
        choices = []
        pattern = r'\(([A-D])\)([^(]*?)(?=\([A-D]\)|$)'
        matches = re.findall(pattern, choices_value)
        for letter, content in matches:
            choices.append(f"({letter}){content.strip()}")

        return choices if choices else [choices_value]

    return [str(choices_value)]


def parse_gold(gold_value) -> List[int]:
    """解析正确答案"""
    if isinstance(gold_value, list):
        return gold_value

    try:
        gold = ast.literal_eval(str(gold_value))
        if isinstance(gold, list):
            return gold
        return [gold]
    except:
        return [0]


def format_question(query: str, choices: List[str], is_multi_answer: bool = False) -> str:
    """格式化问题为标准 prompt"""
    # 清理 query - 提取纯问题部分
    query = query.strip()

    # 移除 "问题：" 前缀
    if query.startswith('问题：'):
        query = query[3:].strip()

    # 移除 query 中已嵌入的选项（从 "选项：" 开始）
    if '选项：' in query:
        query = query.split('选项：')[0].strip()

    # 移除末尾可能的 "答案：..." 部分
    if '答案：' in query:
        query = query.split('答案：')[0].strip()

    # 构建选项
    options = []
    for i, choice in enumerate(choices):
        choice = choice.strip()
        # 移除已有的选项标记 (A), (B) 等
        choice = re.sub(r'^\([A-D]\)\s*', '', choice)
        letter = chr(65 + i)  # A, B, C, D
        options.append(f"{letter}: {choice}")

    # 根据是否多选调整提示
    if is_multi_answer:
        instruction = (
            "请你运用法律知识从A,B,C,D中选出所有正确的答案，不需要说明理由。"
            "请你严格按照这个格式回答：[正确答案]A,B<eoa>（多个答案用逗号分隔）。"
        )
    else:
        instruction = (
            "请你运用法律知识从A,B,C,D中选出一个正确的答案，不需要说明理由。"
            "请你严格按照这个格式回答：[正确答案]A<eoa>。"
        )

    prompt = f"{instruction}\n\n{query}\n\n" + "\n".join(options)

    return prompt


def process_jec_qa(parquet_path: str, dataset_name: str = "jec-qa") -> List[Dict]:
    """处理 JEC-QA parquet 文件"""
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples from {parquet_path}")

    results = []
    single_answer_count = 0
    multi_answer_count = 0

    for idx, row in df.iterrows():
        try:
            query = row['query']
            choices = parse_choices(row['choices'])
            gold = parse_gold(row['gold'])

            # 判断是否多选题
            is_multi_answer = len(gold) > 1

            if is_multi_answer:
                multi_answer_count += 1
            else:
                single_answer_count += 1

            # 格式化 prompt
            correct_prompt = format_question(query, choices, is_multi_answer)
            incorrect_prompt = create_masked_prompt(correct_prompt)

            # 生成答案字符串
            answer_letters = [chr(65 + g) for g in gold]
            answer = ",".join(answer_letters)

            results.append({
                "correct_prompt": correct_prompt,
                "incorrect_prompt": incorrect_prompt,
                "answer": answer,
                "metadata": {
                    "source": dataset_name,
                    "language": "zh",
                    "is_multi_answer": is_multi_answer,
                    "n_correct_answers": len(gold),
                    "original_index": idx
                }
            })

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    print(f"Processed {len(results)} samples")
    print(f"  Single-answer: {single_answer_count}")
    print(f"  Multi-answer: {multi_answer_count}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process JEC-QA dataset")
    parser.add_argument("--data-dir", default="data/jec-qa", help="JEC-QA data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--single-only", action="store_true", help="Only process single-answer questions")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    # 处理 JEC-QA-KD
    kd_path = os.path.join(args.data_dir, "jec-qa-kd.parquet")
    if os.path.exists(kd_path):
        print("\n" + "=" * 50)
        print("Processing JEC-QA-KD (Knowledge-Driven)")
        print("=" * 50)
        kd_results = process_jec_qa(kd_path, "JEC-QA-KD")
        all_results.extend(kd_results)

    # 处理 JEC-QA-CA
    ca_path = os.path.join(args.data_dir, "jec-qa-ca.parquet")
    if os.path.exists(ca_path):
        print("\n" + "=" * 50)
        print("Processing JEC-QA-CA (Case-Analysis)")
        print("=" * 50)
        ca_results = process_jec_qa(ca_path, "JEC-QA-CA")
        all_results.extend(ca_results)

    # 如果只要单选题
    if args.single_only:
        all_results = [r for r in all_results if not r['metadata']['is_multi_answer']]
        print(f"\nFiltered to {len(all_results)} single-answer questions")

    # 保存结果
    output_path = os.path.join(args.output_dir, "chinese_legal.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Total samples: {len(all_results)}")
    print(f"Output: {output_path}")

    # 统计
    single = sum(1 for r in all_results if not r['metadata']['is_multi_answer'])
    multi = len(all_results) - single
    print(f"Single-answer: {single}")
    print(f"Multi-answer: {multi}")


if __name__ == "__main__":
    main()
