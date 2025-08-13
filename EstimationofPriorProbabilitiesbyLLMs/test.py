import os, argparse, re
import pandas as pd
from collections import defaultdict
from EstimationofPriorProbabilitiesbyLLMs.utils.helpers import read_names_from_csv, model_name_translation
from EstimationofPriorProbabilitiesbyLLMs.utils.test_utils import answer_extractor, print_metric_tables, calculate_metrics, calculate_BN_KL_divergence
from tqdm import tqdm
from PreprocessingBayesianNetworks.save_model_pickles import safe_load


llm_models = ['GPT-4o (SepState)', 'GPT-4o (FullDist)', 'GPT-4o (Token Probability)', 'GPT-4o (Without Context)', 'GPT-4o (Random)',\
 'GPT-4o-mini (SepState)', 'GPT-4o-mini (FullDist)', 'GPT-4o-mini (Token Probability)', 'GPT-4o-mini (Without Context)', 'GPT-4o-mini (Random)',\
 'DeekSeek-V3 (SepState)', 'DeekSeek-V3 (FullDist)', 'DeekSeek-V3 (Token Probability)', 'DeekSeek-V3 (Without Context)', 'DeekSeek-V3 (Random)',\
"Gemini Pro (SepState)","Claude 3.5 (SepState)","Gemini Pro (FullDist)","Claude 3.5 (FullDist)",\
 'O3 (SepState)', 'O3 (FullDist)',\
 'O3-mini (SepState)', 'O3-mini (FullDist)',\
 'DeekSeek-R1 (SepState)', 'DeekSeek-R1 (FullDist)',\
 'Uniform', 'Random']

parser = argparse.ArgumentParser(description='Test LLMs for bayesian inference')
parser.add_argument('--outputdataset', dest='outputdataset', default="dataset_outputs/",help='dataset folder to read the results from', type=str)
parser.add_argument('--models',dest='models',nargs='+',default=llm_models, help="Specify one or more models.")
parser.add_argument('--verbose',dest='verbose',default=False, type=bool, help="Prints the results of each model.")

args = parser.parse_args()

args.models = [model_name_translation[name] for name in args.models]

names, info_raw = read_names_from_csv('../PreprocessingBayesianNetworks/bn_node_explanations.csv')
print("Models loading...")
models = {}
for model_name in tqdm(names):
    models[model_name] = safe_load(f"../PreprocessingBayesianNetworks/BNs/Pickles/{model_name}.pkl")
print("Models loaded.\nTesting...")

for cathegorization_factor_name in ["BN Models", "Number of States", "Number of Parents"]:
    results_df, results = [], defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    for model_name in args.models:
        missed_token, total_token = 0, 0
        for data_set_index in range(len(names)):
            print(model_name, names[data_set_index])
            node_names, predicted_answers, true_answers, same_CPT, prev_value  = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), -1
            file_name = f"{args.outputdataset}datasetname_{names[data_set_index]}_model_name_{model_name}.csv"
            if not os.path.exists(file_name):
                print(f"File {file_name} does not exist.")
                continue

            answer_data = pd.read_csv(file_name)[:]
            prev_node = ""
            node_state_count = dict()
            for _, i_LLM in answer_data.iterrows():
                if i_LLM["question"].rsplit(" is ", 1)[0] not in node_state_count:
                    node_state_count[i_LLM["question"].rsplit(" is ", 1)[0]] = 1
                else:
                    node_state_count[i_LLM["question"].rsplit(" is ", 1)[0]] += 1

            for _, i_LLM in answer_data.iterrows():
                try:
                    predicted_answer = answer_extractor(str(i_LLM["raw_text"]))
                    if predicted_answer==0:
                        missed_token+=1
                    total_token+=1
                except:
                    predicted_answer = 0
                    print("\nError: The probability could not be extracted.\n", i_LLM["raw_text"])
                while predicted_answer > 1:
                    print("\nError: The probability is extracted incorrectly.\n", i_LLM["raw_text"])
                    predicted_answer /= 100

                try: true_answer = float(i_LLM["answer"].strip(" .%")) / 100
                except:
                    print("ERROR: Ground Truth not found.")
                    true_answer = 0

                curr_node = i_LLM["question"][:i_LLM["question"].rindex(" is ")].strip()
                cathegorization_factor= (len(set(re.findall(r'\b(\w+)\s+is\b', i_LLM["question"]))) - 1) if args.cathegorization == "by parents" else model_name
                if args.cathegorization == "by states":
                    cathegorization_factor = node_state_count[i_LLM["question"].rsplit(" is ", 1)[0]]
                predicted_answers[cathegorization_factor].append(predicted_answer)
                true_answers[cathegorization_factor].append(true_answer)
                if prev_node == curr_node:
                    same_CPT[cathegorization_factor].append(prev_value)
                    node_names[cathegorization_factor].append(curr_node)
                else:
                    prev_value += 1
                    prev_node = curr_node
                    same_CPT[cathegorization_factor].append(prev_value)
                    node_names[cathegorization_factor].append(curr_node)

            calculate_metrics(predicted_answers,true_answers,same_CPT,results,names,data_set_index,model_name,results_df,node_names,cathegorization_factor_name=cathegorization_factor_name)
            results_df[-1]["BN KL Divergence"] = calculate_BN_KL_divergence(models[names[data_set_index]],file_name)
            results_df[-2]["BN KL Divergence"] = -1
        if "token" in model_name:
            print(f"Missed tokens percentage for {model_name}: {missed_token/total_token*100}%")

    pd.DataFrame(results_df).to_csv(f'results/llm_evaluation_results_{cathegorization_factor_name}.csv', index=False)
    if args.verbose:
        print_metric_tables(results, ["R2 Score", "KL Divergence"],names)