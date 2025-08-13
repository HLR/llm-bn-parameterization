import os, argparse

from EstimationofPriorProbabilitiesbyLLMs.utils.llm import get_llm
from classification.methods.COT import run_COT_method
from classification.methods.Random_method import run_Random_method

from classification.methods.HC_Data import run_HC_Data
from classification.methods.NB_Data import run_NB_Data

from classification.methods.HC_FullDist import run_HC_FullDist
from classification.methods.HC_SepState import run_HC_SepState

from classification.methods.NB_FullDist import run_NB_FullDist
from classification.methods.NB_SepState import run_NB_SepState

from classification.methods.pseudocount import run_pseudocount

ALL_METHODS = [
    "COT", "Random",
    "HC_Data", "HC_FullDist", "HC_SepState", "HC_FullDist_Pseudocount", "HC_SepState_Pseudocount",
    "NB_Data", "NB_FullDist", "NB_SepState", "NB_FullDist_Pseudocount", "NB_SepState_Pseudocount"
]

parser = argparse.ArgumentParser(description='Run different methods for classification using BNs.')
parser.add_argument('--outputdatasets', dest='outputdatasets', default="results", help='Folder to save the result CSV files', type=str)
parser.add_argument('--outputmodels', dest='outputmodels', default="saved_models", help='Folder to save intermediate/model artifacts', type=str)
parser.add_argument('--methods', dest='methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS, help="One or more training methods.")
parser.add_argument('--workers', dest='workers', default=4, help="Number of parallel workers", type=int)
parser.add_argument('--models', dest='models', nargs='+', default=["gpt-4o"], help="One or more chat models (ignored for *Data methods).")
parser.add_argument('--splits', dest='splits', nargs='+', default=[10, 20], type=int, help="Train/validation splits for *Data and *Pseudocount methods. Use -1 for full data.")
parser.add_argument('--runs', dest='runs', default=5, help='How many times to run experiments (applies to *Data and *Pseudocount with split != -1)', type=int)
args = parser.parse_args()

args.outputmodels = os.path.abspath(args.outputmodels)
args.outputdatasets = os.path.abspath(args.outputdatasets)

os.makedirs(args.outputmodels, exist_ok=True)
os.makedirs(args.outputdatasets, exist_ok=True)

for split in args.splits:
    for method in args.methods:
        if method in ["Random", "COT", "HC_FullDist", "HC_SepState", "NB_FullDist", "NB_SepState"] and split != -1:
            continue

        models = args.models
        if "Data" in method:
            models = ["Nollm"]

        for llm in models:
            runs = [0]
            if ("Data" in method or "Pseudocount" in method) and split != -1:
                runs = range(args.runs)

            for run_id in runs:
                chat_model = None
                if llm != "Nollm":
                    chat_model = get_llm(llm)

                result_file_name = os.path.join(args.outputdatasets, f"method_{method}_llm_{llm}_split_{split}_run_{run_id}.csv")
                data_file_folder = os.path.join(args.outputmodels, f"method_{method}_llm_{llm}_split_{split}_run_{run_id}")
                os.makedirs(data_file_folder, exist_ok=True)

                # Upstream artifact roots for dependent methods
                hc_data_root = os.path.join(args.outputmodels, "method_HC_Data_llm_Nollm_split_-1_run_0")
                nb_fulldist_root = os.path.join(args.outputmodels, f"method_NB_FullDist_llm_{llm}_split_-1_run_0")
                nb_SepState_root = os.path.join(args.outputmodels, f"method_NB_SepState_llm_{llm}_split_-1_run_0")
                hc_fulldist_root = os.path.join(args.outputmodels, f"method_HC_FullDist_llm_{llm}_split_-1_run_0")
                hc_SepState_root = os.path.join(args.outputmodels, f"method_HC_SepState_llm_{llm}_split_-1_run_0")

                if method == "COT":
                    run_COT_method(chat_model, result_file_name, NUM_WORKERS=args.workers)
                elif method == "Random":
                    run_Random_method(result_file_name, NUM_WORKERS=args.workers)
                # HC
                elif method == "HC_Data":
                    run_HC_Data(result_file_name, data_file_folder, split, run_id)
                elif method == "HC_FullDist":
                    run_HC_FullDist(chat_model, hc_data_root, result_file_name, data_file_folder)
                elif method == "HC_SepState":
                    run_HC_SepState(chat_model, hc_data_root, result_file_name, data_file_folder)
                elif method == "HC_FullDist_Pseudocount":
                    run_pseudocount(hc_fulldist_root, result_file_name, split, run_id)
                elif method == "HC_SepState_Pseudocount":
                    run_pseudocount(hc_SepState_root, result_file_name, split, run_id)
                # NB
                elif method == "NB_Data":
                    run_NB_Data(result_file_name, data_file_folder, split, run_id)
                elif method == "NB_FullDist":
                    run_NB_FullDist(chat_model, result_file_name, data_file_folder)
                elif method == "NB_SepState":
                    run_NB_SepState(chat_model, result_file_name, data_file_folder)
                elif method == "NB_FullDist_Pseudocount":
                    run_pseudocount(nb_fulldist_root, result_file_name, split, run_id)
                elif method == "NB_SepState_Pseudocount":
                    run_pseudocount(nb_SepState_root, result_file_name, split, run_id)
