# Hybrid Bayesian Estimation

This module combines prior probabilities estimated by LLMs with data sampled from Bayesian Networks (BNs) to learn/update Conditional Probability Tables (CPTs). The main entry point is `main.py`. It iterates over BN models, sampling settings, and learning methods, and writes a summary CSV of metrics. Optionally, it can save BIF files for the original, initial (LLM prior), learned-from-data, and combined models.

After running `main.py`, you can use `analysis_BNandData.py` to generate plots comparing models and sample sizes.

## Usage

Typical workflow:

1) Generate results (comparison_results.csv)
- python main.py --output .\outputs\ --models "GPT-4o (SepState)" "Uniform" --learning_method Counting --sampling_method per_model_by_parameter --prior_weight_hyper_parameter 2 --save_BNs False

Notes:
- Initial CPT priors are read from CSVs produced by the upstream module: `EstimationofPriorProbabilitiesbyLLMs\dataset_outputs\datasetname_<BN_NAME>_model_name_<LLM_ID>.csv`.
- Sampling caches are stored in `NodeSamplingCache\<samples>_<model>_<sampling_method>_shared_samples.pkl` for consistency across runs.

2) Plot analysis
- python analysis_BNandData.py --csv_file .\outputs\comparison_results.csv --model gpt-4o --learning_method counting --metric "BN KL Divergence"

This will show boxplots across models and sample sizes. See the Arguments sections below for more details.

## Command-line arguments (main.py)

python main.py [options]

- --save_BNs (bool, default: False)
  - Whether to write per-BN BIF files with initial/learned/combined CPTs.
- --output (str, default: "./")
  - Output directory for the summary CSV (and optional BIFs). The CSV is written as `<output>\\comparison_results.csv`.
- --learning_method (one or more, choices: Dirichlet, Counting; default: Counting Dirichlet)
  - Dirichlet: linear pooling of probabilities; Counting: treat priors as pseudocounts.
- --prior_weight_hyper_parameter (float, default: 2)
- --sampling_method (str, choices: per_node_and_parents, per_node_and_parents_by_size, per_model_by_parameter; default: per_model_by_parameter)
  - Strategy to generate synthetic samples from the true BN.
- --models (one or more, default: "GPT-4o (SepState)" "Uniform")
