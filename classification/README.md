# Classification Experiments (classification/main.py)

This README explains how to run the Bayesian-network–based classification experiments orchestrated by `classification/main.py`. It covers prerequisites, command-line arguments, available methods, examples, and where outputs are written.

# Prerequtisities

To use the code in this folder, install R (and Rtools if you’re on Windows), then install these R packages: stagedtrees, MBCbook, gRbase, and mlbench.

## Overview

`main.py` runs a matrix of experiments across:
- Methods (HC_* and NB_* families, plus baselines like Random and COT)
- Optional LLM backends (for methods that query LLMs)
- Train/validation split sizes
- Multiple random runs

It saves:
- Per-experiment result CSV files under a results folder (default: `classification/results`)
- Intermediate/model artifacts under a models folder (default: `classification/saved_models`)

After running `main.py`, you can use the `report.py` to see the aggregated evaluation results in a nice table.

## Methods

Pass one or more of the following via `--methods`. The default is to run all.

- Baselines:
  - `Random` – random predictions.
  - `COT` – chain-of-thought style prompting.
Hill Climbing variants:
  - `HC_Data` – data-only estimation; does not call LLMs.
  - `HC_FullDist` – Parameterized Hill Climbing using the FullDist scheme.
  - `HC_SepState` – Parameterized Hill Climbing using the SepState scheme.
  - `HC_FullDist_Pseudocount` – EDP using the FullDist scheme to get the priors.
  - `HC_SepState_Pseudocount` – EDP using the SepState scheme to get the priors.
- NB family (Naive Bayes variants):
  - `NB_Data` – data-only estimation; does not call LLMs.
  - `NB_FullDist` - Parameterized Naive Bayes using the FullDist scheme.
  - `NB_SepState` - Parameterized Naive Bayes using the SepState scheme.
  - `NB_FullDist_Pseudocount` - EDP using the FullDist scheme to get the priors.
  - `NB_SepState_Pseudocount` - EDP using the SepState scheme to get the priors.

Notes:
- Methods containing `Data` ignore `--models` and always use a sentinel model `Nollm` (no LLM calls).
- Methods containing `Pseudocount` consume artifacts from the corresponding upstream method run with `split = -1` (full data), i.e., they expect these directories to exist under `--outputmodels`:
  - HC pseudocount: `method_HC_FullDist_llm_<MODEL>_split_-1_run_0` or `method_HC_SepState_llm_<MODEL>_split_-1_run_0`
  - NB pseudocount: `method_NB_FullDist_llm_<MODEL>_split_-1_run_0` or `method_NB_SepState_llm_<MODEL>_split_-1_run_0`

## Command-line arguments

`python classification/main.py [options]`

- `--outputdatasets` (str, default: `results`)
  - Directory to save result CSV files.
- `--outputmodels` (str, default: `saved_models`)
  - Directory to save intermediate/model artifacts.
- `--methods` (one or more of the names listed above; default: all)
  - Training/evaluation methods to run.
- `--workers` (int, default: 4)
  - Number of parallel workers used by the underlying method functions.
- `--models` (one or more model names, default: `deepseek`)
  - LLM backends for LLM-based methods. Supported values in `EstimationofPriorProbabilitiesbyLLMs/utils/llm.py` include:
    - `gpt-4o`,  `deepseek`, which are tested in the paper. But any model supported by langchain should work.
    - `nollm`, which is a dummy model that does not call LLMs.
  - Ignored for `*Data` methods.
- `--splits` (one or more ints, default: `10 20 -1`)
  - Train/validation split sizes (as percentages) for `*Data` and `*Pseudocount` methods. Use `-1` to indicate full data.; see rules below.
- `--runs` (int, default: 5)
  - Number of repeated runs for `*Data` and `*Pseudocount` when `split != -1`.
