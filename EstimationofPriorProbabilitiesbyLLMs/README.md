# Estimation of Prior Probabilities by LLMs

This module runs Large Language Models to estimate prior probabilities over nodes of a collection of Bayesian Networks (BNs). The main entry point is main.py, which iterates over preprocessed BN datasets, queries different LLMs with structured prompts, and saves the resulting probability estimates to CSV files.

## Usage

First execute main.py to generate results. Then run test.py to evaluate the predictions (e.g., KL divergence). Finally, run analysis_BNs.py to plot the results.

## Arguments (from main.py)
- --outputdataset: Folder to store CSV results. Default: dataset_outputs/ (ensure the folder exists).
- --models: One or more model identifiers (see Supported models and modes in Figure 2 of the paper). Default: a union of built-ins from main.py (gpt/o3/deepseek/gemini/claude/Uniform/Random and variants).
- --maxattempt: Max number of retries after a failed prompt (outer loop around run_inference retries). Default: 10.
- --workers: Number of parallel processes (ProcessPoolExecutor). Default: 1.
- --debug: If set to a non-negative integer, run only that BN index instead of the full set. Default: -1.


Token probabilities depend on the backend returning top_logprobs. The utils/llm.py currently enables logprobs for gpt-4o/gpt-4o-mini and deepseek-chat; other providers may not support it or may require additional configuration.

### 1) Generate LLM outputs (main.py)
Example (from this folder):
- python main.py --outputdataset dataset_outputs\ --models "GPT-4o (SepState)" "DeekSeek-V3 (SepState)" --workers 1
Notes:
- The output folder must exist (e.g., dataset_outputs\).

### 2) Evaluate results (test.py)
What it does:
- Reads the CSVs written by main.py from --outputdataset.
- Extracts probabilities, compares with ground truth CPDs, and computes metrics including BN KL Divergence.
- Writes three summary CSVs under results\: llm_evaluation_results_BN Models.csv, llm_evaluation_results_Number of States.csv, llm_evaluation_results_Number of Parents.csv.

Key arguments:

- --outputdataset: where main.py wrote the raw CSVs (default: dataset_outputs\).
Example:
- python test.py --outputdataset dataset_outputs\ --models "GPT-4o (SepState)" "DeekSeek-V3 (SepState)" --verbose True

### 3) Plot analysis (analysis_BNs.py)
What it does:
- Loads results\llm_evaluation_results_*.csv produced by test.py.
- Produces boxplots of the selected metric per model, optionally split by features.
Key arguments:
- --models: same labels as above; mapped to internal IDs.
- --features: optional facets such as "Number of States" or "Number of Parents".
- --metric: CPT KL Divergence or BN KL Divergence (default).
Examples:
- Overall plot:  python analysis_BNs.py --models "GPT-4o (SepState)" "DeekSeek-V3 (SepState)"
- By feature:    python analysis_BNs.py --models "GPT-4o (SepState)" "DeekSeek-V3 (SepState)" --features "Number of States" "Number of Parents"
