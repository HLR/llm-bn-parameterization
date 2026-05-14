# Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization (TMLR 2026)

This repository accompanies the paper:
“Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization”
We evaluate the ability of modern LLMs to estimate conditional probability tables for Bayesian Networks and introduce Expert-Driven Priors (EDP): a pseudocount fusion that combines LLM-derived priors with data to improve parameter estimation, especially under data scarcity.

![Alt text](Figures/Intro.png 'SepState (Left Panel): For each parent configuration, the LLM is prompted with natural‑language descriptions of the node and its parents and queried once per state. The answers are subsequently normalized into a valid conditional distribution, e.g., (75%, 25%). EDP (Right Panel): The LLM‑derived prior distribution is translated into pseudocounts and fused with empirical counts (C) to give the posterior estimates. This treats the LLM as a probabilistic expert whose influence is controlled by a hyper‑parameter.')

## Requirements

- Use Python version 3.13 or higher.
- Run the following commands to install the required packages:
```bash
pip install -r requirements.txt
```
- Set up API keys for the LLMs you want to use. You can do this by creating a .env file in the `EstimationofPriorProbabilitiesbyLLMs` repository or by setting environment variables directly. The following environment variables are required:
  - OPENAI_API_KEY
  - DEEPSEEK_API_KEY
  - GOOGLE_API_KEY
  - ANTHROPIC_API_KEY

## Usage

### Generate LLM‑derived priors (SepState and FullDist)
- From the repository root:
  ```bash
  cd EstimationofPriorProbabilitiesbyLLMs
  python main.py
  ```
- This step produces prior distributions (conditional probability tables) estimated by the LLM. You will use these in the next step.
- To evaluate the quality of the priors, run the following command:
  ```bash
  python test.py
  python analysis_BNs.py
  ```
- Refer to the README in the `EstimationofPriorProbabilitiesbyLLMs` folder for more details.
  
###  Fuse priors with data using EDP
- Then:
  ```bash
  cd ../EDP
  python main.py
  ```
- This combines empirical counts with the LLM‑derived priors to estimate Bayesian Network parameters under the EDP framework.
- To evaluate the quality of the priors, run the following command:
  ```bash
  python test.py
  python analysis_BNandData.py
  ```
- Refer to the README in the `EDP` folder for more details.

###  Run downstream classification experiments
- Finally:
  ```bash
  cd ../classification
  ```
- Use the scripts in this folder to run classification experiments and evaluate EDP on downstream tasks.
- You need additional setup to run the classification experiments. Please refer to the README in the classification folder.

## Citation

Please cite our paper if you find this code useful:
```
@misc{nafar2025extractingprobabilisticknowledgelarge,
      title={Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization}, 
      author={Aliakbar Nafar and Kristen Brent Venable and Zijun Cui and Parisa Kordjamshidi},
      year={2025},
      eprint={2505.15918},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15918}, 
}
```
