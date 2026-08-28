

import os
import re
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pathlib import Path

class PathPatchingDataProcessor:

    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.random_seed = random_seed

    def create_masked_prompt(self, text: str, mask_ratio: float = 0.3) -> str:
        raise NotImplementedError

    def format_for_path_patching(self, data: Dict) -> Dict:
        raise NotImplementedError

    def save_dataset(self, data: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples to {output_path}")

class LEXamProcessor(PathPatchingDataProcessor):

    LEGAL_TERMS = [

        'court', 'judge', 'jury', 'plaintiff', 'defendant', 'attorney', 'lawyer',
        'prosecutor', 'witness', 'appellant', 'respondent', 'claimant',

        'liability', 'negligence', 'damages', 'breach', 'contract', 'tort',
        'jurisdiction', 'statute', 'regulation', 'precedent', 'remedy',
        'injunction', 'appeal', 'verdict', 'sentence', 'conviction',

        'sue', 'prosecute', 'appeal', 'dismiss', 'overturn', 'affirm',
        'grant', 'deny', 'enjoin', 'stipulate',

        'reasonable', 'proximate', 'foreseeable', 'material', 'substantial',
        'prima facie', 'bona fide', 'de facto', 'de jure',

        'cantonal', 'federal', 'swiss', 'confederation',
    ]

    def __init__(self, parquet_path: str, random_seed: int = 42):
        super().__init__(random_seed)
        self.parquet_path = parquet_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_parquet(self.parquet_path)
        print(f"Loaded {len(self.df)} total samples")
        return self.df

    def filter_english(self) -> pd.DataFrame:
        if self.df is None:
            self.load_data()
        en_df = self.df[self.df['language'] == 'en'].copy()
        print(f"Filtered {len(en_df)} English samples")
        return en_df

    def create_masked_prompt(self, text: str, mask_ratio: float = 0.3) -> str:
        masked_text = text

        terms_found = []
        for term in self.LEGAL_TERMS:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                terms_found.append((match.start(), match.end(), match.group()))

        terms_found.sort(key=lambda x: x[0], reverse=True)

        n_to_mask = max(1, int(len(terms_found) * mask_ratio))
        if terms_found:
            terms_to_mask = random.sample(terms_found, min(n_to_mask, len(terms_found)))

            for start, end, term in terms_to_mask:

                placeholder = f"**{chr(65 + random.randint(0, 25))}**"
                masked_text = masked_text[:start] + placeholder + masked_text[end:]

        return masked_text

    def format_question(self, row: pd.Series) -> str:
        import ast

        question = row['question']
        choices = row['choices']

        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except (ValueError, SyntaxError):

                choices = [choices]

        options = []
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            options.append(f"{letter}: {choice}")

        prompt = (
            "Please use your legal knowledge to select the correct answer from the options below. "
            "Write your answer between [Answer] and <eoa>. For example: [Answer]A<eoa>. "
            "Please strictly follow this format.\n\n"
            f"{question}\n\n"
            + "\n".join(options)
        )

        return prompt

    def format_for_path_patching(self, row: pd.Series) -> Dict:
        correct_prompt = self.format_question(row)
        incorrect_prompt = self.create_masked_prompt(correct_prompt)

        gold_idx = row['gold']
        answer = chr(65 + gold_idx)

        return {
            "correct_prompt": correct_prompt,
            "incorrect_prompt": incorrect_prompt,
            "answer": answer,
            "metadata": {
                "source": "LEXam",
                "id": row.get('id', ''),
                "course": row.get('course', ''),
                "area": row.get('area', ''),
                "jurisdiction": row.get('jurisdiction', ''),
                "year": row.get('year', ''),
                "language": "en"
            }
        }

    def process_all(self, max_samples: Optional[int] = None) -> List[Dict]:
        en_df = self.filter_english()

        if max_samples:
            en_df = en_df.head(max_samples)

        results = []
        for _, row in en_df.iterrows():
            try:
                formatted = self.format_for_path_patching(row)
                results.append(formatted)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        print(f"Successfully processed {len(results)} samples")
        return results

class MoralChoiceProcessor(PathPatchingDataProcessor):

    MORAL_TERMS_EN = [
        'right', 'wrong', 'moral', 'ethical', 'fair', 'unfair', 'just', 'unjust',
        'honest', 'dishonest', 'harm', 'help', 'good', 'bad', 'virtue', 'vice',
        'duty', 'obligation', 'responsibility', 'conscience', 'guilt', 'shame',
        'promise', 'trust', 'betray', 'loyal', 'respect', 'dignity',
    ]

    MORAL_TERMS_ZH = [
        '道德', '伦理', '正确', '错误', '公平', '公正', '诚实', '欺骗',
        '伤害', '帮助', '善良', '邪恶', '义务', '责任', '良心', '内疚',
        '承诺', '信任', '背叛', '忠诚', '尊重', '尊严', '应该', '不应该',
    ]

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)

    def create_masked_prompt(self, text: str, is_chinese: bool = False, mask_ratio: float = 0.3) -> str:
        terms = self.MORAL_TERMS_ZH if is_chinese else self.MORAL_TERMS_EN
        masked_text = text

        terms_found = []
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE if not is_chinese else 0)
            for match in pattern.finditer(text):
                terms_found.append((match.start(), match.end(), match.group()))

        terms_found.sort(key=lambda x: x[0], reverse=True)

        n_to_mask = max(1, int(len(terms_found) * mask_ratio))
        if terms_found:
            terms_to_mask = random.sample(terms_found, min(n_to_mask, len(terms_found)))

            for start, end, term in terms_to_mask:
                placeholder = f"**{chr(65 + random.randint(0, 25))}**"
                masked_text = masked_text[:start] + placeholder + masked_text[end:]

        return masked_text

    def process_english_moral(self, csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} moral choice samples")

        if max_samples:
            df = df.head(max_samples)

        results = []
        for _, row in df.iterrows():
            try:
                context = row.get('context', row.get('original_context', ''))
                action1 = row.get('action1', row.get('original_action1', ''))
                action2 = row.get('action2', row.get('original_action2', ''))

                if not context or not action1 or not action2:
                    continue

                correct_prompt = (
                    "Please make a moral judgment on the following scenario. "
                    "Choose the action you think is more appropriate from A and B. "
                    "Write your answer between [Answer] and <eoa>. For example: [Answer]A<eoa>.\n\n"
                    f"Scenario: {context}\n\n"
                    f"A: {action1}\n"
                    f"B: {action2}"
                )

                incorrect_prompt = self.create_masked_prompt(correct_prompt, is_chinese=False)

                answer = "A"

                results.append({
                    "correct_prompt": correct_prompt,
                    "incorrect_prompt": incorrect_prompt,
                    "answer": answer,
                    "metadata": {
                        "source": "MoralChoice",
                        "scenario_id": row.get('scenario_id', ''),
                        "ambiguity": row.get('ambiguity', ''),
                        "language": "en"
                    }
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        print(f"Successfully processed {len(results)} English moral samples")
        return results

    def process_chinese_moral(self, csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples from {csv_path}")

        if max_samples:
            df = df.head(max_samples)

        results = []
        for _, row in df.iterrows():
            try:

                context = row.get('chinese_context', '')
                action1 = row.get('chinese_action1', '')
                action2 = row.get('chinese_action2', '')

                if re.search(r'[a-zA-Z]{3,}', context):

                    context = row.get('original_context', context)
                    action1 = row.get('original_action1', action1)
                    action2 = row.get('original_action2', action2)

                    needs_translation = True
                else:
                    needs_translation = False

                if not context or not action1 or not action2:
                    continue

                correct_prompt = (
                    "请对以下情境做出道德判断。从A、B中选择你认为更合适的行为，"
                    "并将答案写在[正确答案]和<eoa>之间。例如：[正确答案]A<eoa>。\n\n"
                    f"情境：{context}\n\n"
                    f"A：{action1}\n"
                    f"B：{action2}"
                )

                incorrect_prompt = self.create_masked_prompt(correct_prompt, is_chinese=True)

                answer = "A"

                results.append({
                    "correct_prompt": correct_prompt,
                    "incorrect_prompt": incorrect_prompt,
                    "answer": answer,
                    "metadata": {
                        "source": "MoralChoice_Chinese",
                        "scenario_id": row.get('scenario_id', ''),
                        "ambiguity": row.get('ambiguity', ''),
                        "language": "zh",
                        "needs_translation": needs_translation
                    }
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        print(f"Successfully processed {len(results)} Chinese moral samples")
        return results

class GeneralMCQAProcessor(PathPatchingDataProcessor):

    GENERAL_TERMS = [

        'force', 'energy', 'momentum', 'velocity', 'acceleration', 'mass',
        'gravity', 'electric', 'magnetic', 'wave', 'frequency', 'photon',

        'molecule', 'atom', 'electron', 'proton', 'neutron', 'bond',
        'reaction', 'solution', 'acid', 'base', 'oxidation', 'reduction',

        'equation', 'function', 'integral', 'derivative', 'matrix', 'vector',
        'theorem', 'proof', 'probability', 'limit', 'convergence',

        'cell', 'protein', 'gene', 'DNA', 'RNA', 'enzyme', 'organism',
        'mutation', 'evolution', 'metabolism', 'mitosis',

        'algorithm', 'complexity', 'binary', 'compiler', 'memory', 'processor',
    ]

    HUMANITIES_TERMS = [

        'philosophy', 'ethics', 'metaphysics', 'epistemology', 'ontology',
        'existentialism', 'rationalism', 'empiricism', 'utilitarianism',

        'empire', 'dynasty', 'revolution', 'treaty', 'colony', 'republic',
        'monarchy', 'democracy', 'war', 'civilization', 'conquest',

        'religion', 'theology', 'scripture', 'doctrine', 'faith', 'worship',
        'salvation', 'divine', 'sacred', 'ritual', 'prophecy',

        'culture', 'tradition', 'ideology', 'movement', 'era', 'century',
    ]

    STEM_SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'college_biology',
        'college_chemistry', 'college_computer_science', 'college_mathematics',
        'college_physics', 'computer_security', 'conceptual_physics',
        'electrical_engineering', 'elementary_mathematics',
        'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics',
        'high_school_physics', 'high_school_statistics',
        'machine_learning', 'college_medicine',
    ]

    HUMANITIES_SUBJECTS = [
        'philosophy', 'world_religions', 'high_school_european_history',
        'high_school_us_history', 'high_school_world_history',
        'prehistory', 'moral_disputes', 'moral_scenarios',
        'logical_fallacies', 'formal_logic', 'jurisprudence',
        'international_law', 'professional_law',
        'high_school_geography', 'sociology', 'us_foreign_policy',
        'high_school_government_and_politics', 'public_relations',
        'human_aging', 'human_sexuality',
    ]

    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'and', 'but', 'or', 'if', 'it',
        'its', 'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we',
        'you', 'i', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
        'their', 'our', 'which', 'who', 'whom', 'what', 'about',
    }

    PLACEHOLDER_TYPES = {
        'letter': lambda self: f"**{chr(65 + random.randint(0, 25))}**",
        'mask': lambda self: "[MASK]",
        'random_word': lambda self: random.choice(['item', 'thing', 'object', 'element', 'unit']),
        'empty': lambda self: "",
    }

    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)

    def _get_content_words(self, text: str) -> List[Tuple[int, int, str]]:
        content_words = []

        for match in re.finditer(r'\b([a-zA-Z]{3,})\b', text):
            word = match.group(1).lower()
            if word not in self.STOP_WORDS:
                content_words.append((match.start(), match.end(), match.group()))
        return content_words

    def create_masked_prompt(self, text: str, mask_ratio: float = 0.3,
                             placeholder_type: str = 'letter',
                             domain_terms: Optional[List[str]] = None) -> str:
        if domain_terms is None:
            domain_terms = self.GENERAL_TERMS

        masked_text = text

        terms_found = []
        for term in domain_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                terms_found.append((match.start(), match.end(), match.group()))

        if not terms_found:
            content_words = self._get_content_words(text)
            if content_words:
                n_to_mask = max(1, int(len(content_words) * mask_ratio))
                terms_found = random.sample(content_words, min(n_to_mask, len(content_words)))

        terms_found.sort(key=lambda x: x[0], reverse=True)

        n_to_mask = max(1, int(len(terms_found) * mask_ratio))
        if terms_found:
            terms_to_mask = random.sample(terms_found, min(n_to_mask, len(terms_found)))
            for start, end, term in terms_to_mask:
                if placeholder_type == 'mask':
                    placeholder = "[MASK]"
                elif placeholder_type == 'random_word':
                    placeholder = random.choice(['item', 'thing', 'object', 'element', 'unit'])
                elif placeholder_type == 'empty':
                    placeholder = ""
                else:
                    placeholder = f"**{chr(65 + random.randint(0, 25))}**"
                masked_text = masked_text[:start] + placeholder + masked_text[end:]

        return masked_text

    def _process_mmlu_subset(self, json_path: str, subjects: List[str],
                             max_samples: Optional[int] = 500,
                             source_label: str = "MMLU_STEM",
                             mask_ratio: float = 0.3,
                             placeholder_type: str = 'letter',
                             domain_terms: Optional[List[str]] = None) -> List[Dict]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        filtered_data = [d for d in data if d.get('subject', '') in subjects]
        print(f"MMLU total: {len(data)}, filtered ({source_label}): {len(filtered_data)}")

        if max_samples and len(filtered_data) > max_samples:
            filtered_data = random.sample(filtered_data, max_samples)

        results = []
        identical_count = 0
        for item in filtered_data:
            try:
                question = item['question']
                choices = item['choices']
                answer_idx = item['answer']

                options = []
                for i, choice in enumerate(choices):
                    letter = chr(65 + i)
                    options.append(f"{letter}: {choice}")

                correct_prompt = (
                    "Please select the correct answer from the options below. "
                    "Write your answer between [Answer] and <eoa>. For example: [Answer]A<eoa>. "
                    "Please strictly follow this format.\n\n"
                    f"{question}\n\n"
                    + "\n".join(options)
                )

                incorrect_prompt = self.create_masked_prompt(
                    correct_prompt,
                    mask_ratio=mask_ratio,
                    placeholder_type=placeholder_type,
                    domain_terms=domain_terms
                )

                if correct_prompt == incorrect_prompt:
                    identical_count += 1

                if isinstance(answer_idx, int):
                    answer = chr(65 + answer_idx)
                else:
                    answer = str(answer_idx).strip().upper()

                results.append({
                    "correct_prompt": correct_prompt,
                    "incorrect_prompt": incorrect_prompt,
                    "answer": answer,
                    "metadata": {
                        "source": source_label,
                        "subject": item.get('subject', ''),
                        "language": "en"
                    }
                })
            except Exception as e:
                continue

        if identical_count > 0:
            print(f"WARNING: {identical_count}/{len(results)} samples have identical correct/incorrect prompts")
        print(f"Successfully processed {len(results)} {source_label} samples")
        return results

    def process_mmlu_data(self, json_path: str, max_samples: Optional[int] = 500,
                          mask_ratio: float = 0.3,
                          placeholder_type: str = 'letter') -> List[Dict]:
        return self._process_mmlu_subset(
            json_path=json_path,
            subjects=self.STEM_SUBJECTS,
            max_samples=max_samples,
            source_label="MMLU_STEM",
            mask_ratio=mask_ratio,
            placeholder_type=placeholder_type,
            domain_terms=self.GENERAL_TERMS
        )

    def process_mmlu_humanities(self, json_path: str, max_samples: Optional[int] = 500,
                                mask_ratio: float = 0.3,
                                placeholder_type: str = 'letter') -> List[Dict]:
        return self._process_mmlu_subset(
            json_path=json_path,
            subjects=self.HUMANITIES_SUBJECTS,
            max_samples=max_samples,
            source_label="MMLU_HUMANITIES",
            mask_ratio=mask_ratio,
            placeholder_type=placeholder_type,
            domain_terms=self.HUMANITIES_TERMS
        )

class ChineseLawProcessor(PathPatchingDataProcessor):

    LEGAL_TERMS_ZH = [
        '法律', '法规', '条例', '规定', '民法', '刑法', '宪法', '行政法',
        '诉讼', '起诉', '上诉', '判决', '裁定', '执行', '强制', '管辖',
        '原告', '被告', '当事人', '代理人', '律师', '法官', '检察官',
        '权利', '义务', '责任', '赔偿', '违约', '侵权', '犯罪', '处罚',
        '合同', '协议', '合法', '违法', '有效', '无效', '撤销', '解除',
    ]

    def __init__(self, json_path: str, random_seed: int = 42):
        super().__init__(random_seed)
        self.json_path = json_path

    def load_data(self) -> List[Dict]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        try:
            data = json.loads(content)
            if isinstance(data, list):
                print(f"Loaded {len(data)} Chinese law samples (JSON array)")
                return data
        except json.JSONDecodeError:
            pass

        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(data)} Chinese law samples (JSONL)")
        return data

    def create_masked_prompt(self, text: str, mask_ratio: float = 0.3) -> str:
        masked_text = text

        terms_found = []
        for term in self.LEGAL_TERMS_ZH:
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

    def validate_and_fix(self, data: List[Dict]) -> List[Dict]:
        valid_data = []

        for item in data:

            if 'correct_prompt' not in item or 'answer' not in item:
                continue

            if 'incorrect_prompt' not in item or not item['incorrect_prompt']:
                item['incorrect_prompt'] = self.create_masked_prompt(item['correct_prompt'])

            answer = item['answer'].strip().upper()
            if len(answer) == 1 and answer in 'ABCD':
                item['answer'] = answer
            else:
                continue

            if 'metadata' not in item:
                item['metadata'] = {
                    "source": "Chinese_Law_Exam",
                    "language": "zh"
                }

            valid_data.append(item)

        print(f"Validated {len(valid_data)} / {len(data)} samples")
        return valid_data

def process_all_datasets(data_dir: str, output_dir: str):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "datasets": {}
    }

    print("\n" + "=" * 50)
    print("Processing LEXam English Legal Data")
    print("=" * 50)

    lexam_path = data_dir / "law" / "test-00000-of-00001.parquet"
    if lexam_path.exists():
        processor = LEXamProcessor(str(lexam_path))
        lexam_data = processor.process_all()
        output_path = output_dir / "english_legal.json"
        processor.save_dataset(lexam_data, str(output_path))
        summary["datasets"]["english_legal"] = {
            "count": len(lexam_data),
            "path": str(output_path)
        }
    else:
        print(f"LEXam file not found: {lexam_path}")

    print("\n" + "=" * 50)
    print("Processing English Moral Data")
    print("=" * 50)

    moral_processor = MoralChoiceProcessor()
    moral_raw_path = data_dir / "moral" / "moralchoice_combined_raw.csv"
    if moral_raw_path.exists():
        en_moral_data = moral_processor.process_english_moral(str(moral_raw_path))
        output_path = output_dir / "english_moral_unbalanced.json"
        moral_processor.save_dataset(en_moral_data, str(output_path))
        summary["datasets"]["english_moral"] = {
            "count": len(en_moral_data),
            "path": str(output_path)
        }

    print("\n" + "=" * 50)
    print("Processing Chinese Moral Data")
    print("=" * 50)

    zh_moral_path = data_dir / "moral" / "moralchoice_chinese_100.csv"
    if zh_moral_path.exists():
        zh_moral_data = moral_processor.process_chinese_moral(str(zh_moral_path))
        output_path = output_dir / f"chinese_moral_{len(zh_moral_data)}_{timestamp}.json"
        moral_processor.save_dataset(zh_moral_data, str(output_path))
        summary["datasets"]["chinese_moral"] = {
            "count": len(zh_moral_data),
            "path": str(output_path)
        }

    print("\n" + "=" * 50)
    print("Processing MMLU Humanities Data")
    print("=" * 50)

    mcqa_processor = GeneralMCQAProcessor()
    mmlu_path = data_dir / "general" / "mmlu_test.jsonl"
    if mmlu_path.exists():

        stem_data = mcqa_processor.process_mmlu_data(str(mmlu_path), max_samples=500)
        output_path = output_dir / "general_mcqa_stem.json"
        mcqa_processor.save_dataset(stem_data, str(output_path))
        summary["datasets"]["general_mcqa_stem"] = {
            "count": len(stem_data),
            "path": str(output_path)
        }

        humanities_data = mcqa_processor.process_mmlu_humanities(str(mmlu_path), max_samples=500)
        output_path = output_dir / "general_mcqa_humanities.json"
        mcqa_processor.save_dataset(humanities_data, str(output_path))
        summary["datasets"]["general_mcqa_humanities"] = {
            "count": len(humanities_data),
            "path": str(output_path)
        }

    print("\n" + "=" * 50)
    print("Validating Chinese Legal Data")
    print("=" * 50)

    zh_law_path = data_dir / "law" / "data.json"
    if zh_law_path.exists():
        processor = ChineseLawProcessor(str(zh_law_path))
        data = processor.load_data()
        valid_data = processor.validate_and_fix(data)
        output_path = output_dir / "chinese_legal.json"
        processor.save_dataset(valid_data, str(output_path))
        summary["datasets"]["chinese_legal"] = {
            "count": len(valid_data),
            "path": str(output_path)
        }

    summary_path = output_dir / f"processing_summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Processing Summary")
    print("=" * 50)
    for dataset, info in summary["datasets"].items():
        print(f"  {dataset}: {info['count']} samples")
    print(f"\nSummary saved to: {summary_path}")

    return summary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process datasets for Path Patching")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    process_all_datasets(args.data_dir, args.output_dir)
