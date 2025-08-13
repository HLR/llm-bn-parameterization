import os,time,random, argparse,sys, math
sys.path.append("../")
from EstimationofPriorProbabilitiesbyLLMs.utils.llm import get_llm
import pandas as pd
from langchain.schema import SystemMessage, HumanMessage
from EstimationofPriorProbabilitiesbyLLMs.utils.helpers import read_names_from_csv, create_text, extract_dict_from_text, prepare_IO, save_llm_answer,answer_extractor,tuple_extractor,model_name_translation
from concurrent.futures import ProcessPoolExecutor
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

parser = argparse.ArgumentParser(description='Run LLMs for Bayesian inference')
parser.add_argument('--outputdataset', dest='outputdataset', default="dataset_outputs/",help='dataset folder to save the results', type=str)
parser.add_argument('--models',dest='models',nargs='+',default=llm_models, help="Specify one or more models.")
parser.add_argument('--maxattempt', dest='maxattempt', default=10,help='Number of tries to get a response from Langchain', type=int)
parser.add_argument('--workers', dest='workers', default=1, type=int, help='Number of parallel workers')
parser.add_argument('--debug', dest='debug', default=-1, type=int, help='Debug mode which determines the number of the Bayesian network to run')
args = parser.parse_args()

args.models = [model_name_translation[name] for name in args.models]

names, info_raw = read_names_from_csv('../PreprocessingBayesianNetworks/bn_node_explanations.csv')
print("Models loading...")
models = {}
for model_name in tqdm(names):
    models[model_name] = safe_load(f"../PreprocessingBayesianNetworks/BNs/Pickles/{model_name}.pkl")
print("Models loaded.\n")

def run_inference(model_name,dataset,info_text):

    file_name = f"{args.outputdataset}datasetname_{dataset}_model_name_{model_name}.csv"
    data, node_possible_states_dict = create_text(models[dataset])
    info = extract_dict_from_text(info_text)

    if os.path.exists(file_name): existing_df = pd.read_csv(file_name)
    else: existing_df = pd.DataFrame()

    answer_mean=sum([float(d.split(" with probability of ")[1].strip(".% "))/100 for d in data])/len(data)
    saved_results=dict()
    for num, i in enumerate(data):
        if num < len(existing_df): continue
        if i.split('with probability of')[0] in saved_results:
            save_llm_answer(file_name, saved_results[i.split('with probability of')[0]], i)
            continue
        saved_results=dict()
        print(model_name, dataset, num)
        question, instruction, node_name = prepare_IO(i,info,"-withoutcontext" in model_name,"-random" in model_name,"-tokenprob" in model_name,"-inoneprompt" in model_name)
        current_node_states = node_possible_states_dict[node_name]
        current_state = i.split('with probability of')[0].rsplit(" is ", 1)[1].strip()

        messages = [
            SystemMessage(content=instruction + "\n Order of states: " + str(node_possible_states_dict[node_name]) +"\n" if "-inoneprompt" in model_name else instruction),
            HumanMessage(content=question if "-tokenprob" in model_name or "-inoneprompt" in model_name else question.replace(" then "," then what is the probability that "))
        ]
        if model_name == "Mean":
            content = answer_mean
        elif model_name == "Random":
            content = random.random()
        else:
            for j in range(args.max_try):
                try:
                    result=get_llm(model_name.split("-withoutcontext")[0].split("-random")[0].split("-tokenprob")[0].split("-inoneprompt")[0]).invoke(messages)
                    if not "-tokenprob" in model_name and not "-inoneprompt" in model_name:
                        if hasattr(result, "content"):
                            content = result.content
                        else:
                            content = result
                        answer_extractor(content)
                    if "-inoneprompt" in model_name:
                        if hasattr(result, "content"):
                            content = result.content
                        else:
                            content = result
                        distributions_from_tuple = tuple_extractor(content)
                        assert len(distributions_from_tuple)==len(current_node_states)
                    break
                except:
                    print(f"Attempt {j + 1} failed.")
                    time.sleep(2)

        if "-tokenprob" in model_name:
            state_token_prob_dict = {i['token']: math.exp(i['logprob']) for i in result.response_metadata['logprobs']['content'][0]['top_logprobs'] }
            for state in current_node_states:
                if state.lower() in state_token_prob_dict:
                    saved_results[(" "+state+" ").join(i.split('with probability of')[0].rsplit(" "+current_state+" ",1) )] = state_token_prob_dict[state.lower()]
                else:
                    saved_results[i.split('with probability of')[0]] = 0
            content = saved_results[i.split('with probability of')[0]]

        if "-inoneprompt" in model_name:
            for num, state in enumerate(current_node_states):
                saved_results[(" "+state+" ").join(i.split('with probability of')[0].rsplit(" "+current_state+" ",1) )] = distributions_from_tuple[num]
            content = saved_results[i.split('with probability of')[0]]

        save_llm_answer(file_name, content, i)

def inference_loop(start, end):
    for i in range(start, end):
        for model_name in args.models:
            for attempt in range(args.maxattempt):
                try:
                    print(f"Running inference for {model_name} on {names[i]}")
                    info_i = i
                    if "-random" in model_name:
                        info_i = random.randint(0, len(info_raw) - 1)
                    run_inference(model_name, names[i], info_raw[info_i])
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
                else: break

if __name__ == "__main__":
    total_items = 80
    num_workers = args.workers
    if num_workers == 1 or not args.debug==-1:
        if args.debug==-1:
            inference_loop(0, total_items)
        else:
            inference_loop(args.debug, args.debug+1)
    else:
        chunk_size = (total_items + num_workers - 1) // num_workers
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for w in range(num_workers):
                start = w * chunk_size
                end = min(start + chunk_size, total_items)
                if start >= end:
                    break
                executor.submit(inference_loop, start, end)
