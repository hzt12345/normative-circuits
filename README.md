# Normative Circuits in LLMs

Reproduction code for the EMNLP 2026 paper *Where Does an LLM Decide What's Right? Sparse Normative Circuits Across Languages and Model Families* (Huang et al., 2026).

## Setup

```bash
python -m pip install -r requirements.txt
```

The experiments use the following Hugging Face checkpoints:

- `Qwen/Qwen3-8B`
- `meta-llama/Llama-3.1-8B`
- `mistralai/Mistral-7B-v0.1`

Llama access requires accepting the model license and authenticating with Hugging Face. Each model requires approximately 20GB or more of GPU memory in FP16. The reported experiments ran on an NVIDIA H800 80GB GPU.

Optional local model paths can be supplied with `QWEN3_8B_PATH`, `LLAMA31_8B_PATH`, and `MISTRAL_7B_V01_PATH`.

## Repository layout

All scripts live in `code/` and run from the repository root. They import each other as
siblings, so the flat layout is intentional.

- **Pipeline** — `unified_path_patching.py` (locate), `set_difference_analysis.py`
  (disentangle), `resampling_knockout.py`, `k_sweep_knockout.py`,
  `sufficiency_experiment.py`, `layer_matched_results.py` (intervene),
  `cross_domain_head_analysis.py` (overlap analysis).
- **Supplementary analyses** — `attention_pattern_analysis.py`, `mlp_analysis.py`,
  `bootstrap_jaccard_analysis.py`, `threshold_sensitivity.py`,
  `sufficiency_effect_size.py`, `run_knockout_validation.py`.
- **Shared modules** — `local_model_loader.py`, `data_processors.py`,
  `head_suppressor.py`, `head_analyzer.py`, `head_sampling.py`, `batch_runner.py`,
  `statistical_tests.py`, `reproducibility.py`.
- **Preprocessing** — `process_jec_qa.py`, `convert_ethics_data.py`,
  `balance_en_moral.py`.

## Benchmark inputs

The paper uses JEC-QA, LEXam, SafetyBench (Ethics and Morality), MoralChoice, and MMLU-STEM. Place the upstream benchmark files under `data/` and run the preprocessing scripts to create these canonical inputs:

```text
data/processed/chinese_legal.json
data/processed/english_legal.json
data/processed/chinese_moral.json
data/processed/english_moral.json
data/processed/general_mcqa_stem.json
```

Each processed file is a JSON list with the fields `correct_prompt`, `incorrect_prompt`, `answer`, and `metadata`. The source benchmarks are available from:

| Dataset | Source |
|---|---|
| JEC-QA | https://jecqa.thunlp.org/ |
| LEXam | https://huggingface.co/datasets/LEXam-Benchmark/LEXam |
| SafetyBench | https://github.com/thu-coai/SafetyBench |
| MoralChoice | https://huggingface.co/datasets/ninoscherrer/moralchoice |
| MMLU | https://huggingface.co/datasets/cais/mmlu |

Expected preprocessing locations are documented in the scripts. A typical preparation sequence is:

```bash
python code/process_jec_qa.py --data-dir data/jec-qa --output-dir data/processed
python code/data_processors.py --data-dir data --output-dir data/processed
python code/convert_ethics_data.py
python code/balance_en_moral.py
```

## Reproducing the main results

Set `MODEL` to one of the checkpoint names above and repeat the commands for all three models. `MODEL_SHORT` is the checkpoint's lower-case final component, such as `qwen3-8b`.

### 1. Locate: path patching

Run the five datasets individually:

```bash
python code/unified_path_patching.py \
  --dataset chinese_legal \
  --model MODEL \
  --method atp
```

Repeat with `english_legal`, `chinese_moral`, `english_moral`, and `general_mcqa`.

### 2. Disentangle: set difference

```bash
python code/set_difference_analysis.py \
  --results-dir output/MODEL_SHORT
```

### 3. Necessity: fixed-K knockout

```bash
python code/resampling_knockout.py \
  --all-datasets \
  --model MODEL \
  --max-samples 500 \
  --n-random-trials 5 \
  --ablation mean \
  --random-control global
```

For the exact per-layer-matched random control:

```bash
python code/resampling_knockout.py \
  --all-datasets \
  --model MODEL \
  --max-samples 500 \
  --n-random-trials 5 \
  --ablation mean \
  --random-control layer_matched

python code/layer_matched_results.py --root .
```

### 4. Sufficiency: amplification

```bash
python code/sufficiency_experiment.py \
  --all-datasets \
  --model MODEL \
  --max-samples 500
```

### 5. K-sweep

```bash
python code/k_sweep_knockout.py \
  --all-datasets \
  --model MODEL \
  --max-samples 500 \
  --n-random-trials 5
```

Experiment artifacts are written under `output/<model>/`.

## Statistical conventions

- Necessity uses logit difference as the primary metric. The reported criterion is a correctly directed NS-vs-random gap with `|delta logit-difference| >= 0.05` and trial-level `|s| >= 2`.
- Sufficiency compares recovery after NS-head amplification against five size-matched random head sets.
- Accuracy intervals use Wilson 95% confidence intervals; random-control summaries report the mean and standard deviation over five trials.

## License

The code is released under the MIT License. Benchmark datasets retain their upstream licenses.
