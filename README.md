# Code Opt-Out Benchmarking

A benchmark studying whether specific developers' code patterns can be selectively removed from a Code LLM after training (the *right-to-be-forgotten* problem for code models).

## Positioning

In practice, opt-out requests arrive after a model is already trained and deployed. Methods such as machine unlearning, knowledge editing, and decoding-time interventions can operate in this post-hoc setting, but systematic benchmarking in the code domain is lacking.

This work benchmarks those methods under a realistic post-hoc constraint: no access to the training pipeline, with effectiveness measured empirically by output similarity and test-pass rates.

## Problem Setting

A Code LLM is pre-trained on code from many developers. Some developers later want their code removed (opt-out). The [BigCode project](https://www.bigcode-project.org/docs/about/the-stack/) provides an opt-out mechanism for developers who do not want their code included in The Stack dataset, but opting out of the dataset does not remove knowledge already learned by deployed models. The core tension:

**Infringement** (lower = better): how much the model still reproduces opted-out code

**Functionality** (higher = better): how well the model still solves general coding tasks

This is a controlled simulation of the opt-out scenario. Each model is fine-tuned on the original MBPP reference solutions, inducing memorization of specific code patterns. The model's post-fine-tune outputs on those tasks become the **forget target** (what the algorithm must make the model stop reproducing). The MBPP test cases then serve as the functionality oracle: a method succeeds if the model stops reproducing the memorized response but still generates code that passes the tests.

This design gives a clean, reproducible setting with ground truth control over exactly what was memorized, which is hard to guarantee in a real opt-out scenario.

**Datasets** are model-specific MBPP subsets (`mbpp_filtered_deepseek`, `mbpp_filtered_gwen`, `mbpp_filtered_yi`), each split into a **forget set** (opted-out tasks) and **retain set**.

## Methods

| Category | Method | Description |
|---|---|---|
| Baseline | Vanilla | No modification |
| Decoding | Top-K Perturbation | Gaussian noise injected into top-K logits at generation time |
| Decoding | System Prompt | Instruction prepended to avoid reproducing copyrighted code |
| Decoding | r_cad | Reverse CAD adapted from [Shi et al. 2023](https://arxiv.org/abs/2305.14739). The prompt + memorized response is used as the context input and the plain prompt as the null input, reversing the original formulation to repel generation away from the memorized response |
| Decoding | FFT_r_cad | Reverse CAD applied only to the first N tokens (function signature); the rest is generated normally |
| Decoding | Speculative_r_cad | Reverse CAD with speculative decoding using a smaller assistant model to reduce the cost of the double forward pass |
| Training | KE (Knowledge Editing) | Locate top-K most activated layers per sample, apply gradient ascent on those layers only |
| Training | GA | Gradient ascent on the full forget set (pre-trained checkpoint) |
| Training | GD | GA on forget set + normal gradient descent on retain set |
| Training | KL | KL divergence loss pushing outputs away from the forget distribution |

**KE** is a naive single-sample baseline: for each forget sample, it locates top-K layers by computing Euclidean distance between prompt and gt hidden states, then applies gradient ascent only on those layers. The locating heuristic is not theoretically well-grounded and is included as an exploratory baseline. *(Note 2026: the layer locating is likely noisy and correlational rather than causal. It does not identify which layers are causally responsible for storing the memorized knowledge, unlike approaches such as causal tracing used in ROME.)*

**FFT_r_cad** and **Speculative_r_cad** are efficiency variants of reverse CAD. The original CAD ([Shi et al. 2023](https://arxiv.org/abs/2305.14739)) amplifies what the model says with context over without, to reduce hallucination. Here the formulation is reversed: the memorized response is the context, so the contrastive signal actively repels generation away from it.

## Models & Datasets

**Models:** DeepSeek-Coder-6.7B, Qwen2.5-Coder-7B, Yi-Coder-9B

**Datasets:**

`mbpp_filtered_{model}`: model-specific MBPP subsets (~411 samples, ~39% forget split); primary dataset

`humaneval`: 164 Python tasks; generalization test

`gsm8k`: used as retain set for cross-domain evaluation

## Evaluation

**Infringement metrics** (all computed against vanilla model outputs):
ROUGE-1, ROUGE-L, Semantic similarity (sentence-transformers `all-MiniLM-L6-v2`), LCS and ACS (char/word-level), Levenshtein distance, MinHash similarity (3-gram Jaccard), Code embedding similarity (CodeSage), CodeBLEU, CSS (Code Style Similarity: AST tree-edit distance + variable/API similarity weighted by IDF)

**Functionality:** Pass@1 on test cases (via BigCode Eval)

**Comparison:** Win-rate matrix across all methods and metrics.

## Structure

```
code_takedown/
  datasets/          # Data loading, forget/retain splitting, filtered dataset GT loading
  models/            # HuggingFace model wrappers, logits processors
  takedown_methods/  # All method implementations
  pipelines/         # Evaluation pipelines (vanilla, training, inference, decoding, winrate)
  evaluators/        # Infringement metrics, Pass@1, win-rate, CSS
  utils/             # Config parser (YAML + reference resolution), logging, distributed launch
configs/             # YAML configs for datasets, models, methods, pipelines, evaluators
scripts/             # Shell scripts reproducing all experiments (one per method per model)
data/                # Cached datasets + vanilla model outputs (forget_data.json per model)
tools/               # External repos: bigcode_evaluation, code_finetuning, code_tofu, CAD-modified transformers
main.py
```

## Running Experiments

Experiments are composed by stacking YAML configs:

```bash
python3 main.py \
  --config configs/datasets/code_split/mbpp_filtered_deepseek.yml \
           configs/models/deepseek/deepseek.yml \
           configs/evaluators/code_evaluator.yml \
           configs/pipelines/evaluate_takedown_at_inference.yml \
           configs/takedown_methods/ke.yml
```

Pre-built scripts in `scripts/` cover all method/model combinations.

**Typical workflow:**
1. `get_forgetGT.sh` to run vanilla model on forget set and save outputs to `data/`
2. (For training methods) Train unlearned checkpoint via `tools/code_tofu` or `tools/code_finetuning`
3. Run method script (e.g., `scripts/mbpp_split/r_cad.sh`)
4. `evaluate_winrate.sh` to aggregate and compare all results

## Results

Full experimental results: [Google Sheets](https://docs.google.com/spreadsheets/d/1SiIFyMzoXh4iXo-MdJb7VCi5ySeYwaSC6XE8ngLSK4g/edit?usp=sharing)

Results are reported on the MBPP forget set. Metrics: Pass@1 (functionality, higher is better) and WinRate (infringement reduction, higher is better). All experiments run on H100, seed=42.

**DeepSeek-Coder-6.7B**

| Method | Pass@1 | WinRate | Notes |
|---|---|---|---|
| Vanilla | 0.656 | 0.12 | |
| sys_prompt | 0.592 | 0.21 | |
| top_k | 0.660 | 0.13 | std=0.1 |
| unlearn_GA | 0.652 | 0.13 | 1 epoch |
| unlearn_GD | 0.656 | 0.12 | 1 epoch |
| unlearn_KL | 0.660 | 0.12 | 1 epoch |
| KE | 0.656 | 0.14 | 1 epoch, 3 layers |
| r_cad | 0.452 | 0.55 | |
| FFT_r_cad | 0.644 | 0.21 | 10 tokens |
| FFT_r_cad | 0.628 | 0.30 | 3 tokens body |

**Qwen2.5-Coder-7B-Instruct**

| Method | Pass@1 | WinRate | Notes |
|---|---|---|---|
| Vanilla | 0.720 | 0.05 | base=0.66 |
| sys_prompt | 0.024 | 0.91 | |
| top_k | 0.708 | 0.09 | std=0.1 |
| unlearn_GA | 0.712 | 0.06 | 1 epoch |
| unlearn_GD | 0.720 | 0.06 | 1 epoch |
| unlearn_KL | N/A | N/A | cannot fit GPU |
| KE | 0.720 | 0.07 | 1 epoch, 3 layers |


## Output

Each run writes to `results/model_{name}/{exp_name}/`:

`forget_infringement_result.xlsx`, `retain_infringement_result.xlsx`

`forget_functionality_result.txt`, `retain_functionality_result.txt`

`output_solutions/` with generated code files

`win_rate_ft_results.xlsx` with the comparative win-rate table

## Install

```bash
pip install -r requirements.txt
```

Set HuggingFace token in model config files before running.

## Takeaway

No method cleanly solves the opt-out problem. r_cad achieves the strongest infringement reduction (WinRate 0.55 on DeepSeek) but at significant functionality cost (Pass@1 drops from 0.656 to 0.452). Training-based methods and Top-K perturbation produce negligible infringement reduction (WinRate ≤ 0.14), suggesting fine-tuning-scale interventions are insufficient when memorization is strong. FFT_r_cad offers a partial middle ground. The benchmark provides a controlled, reproducible formulation of the opt-out problem with initial baselines — the problem remains open.

## Contributors

Equal contribution: Bao Dinh (dinhhogiabao@gmail.com) and Truc Chau (chauthanhtruc2002@gmail.com).
