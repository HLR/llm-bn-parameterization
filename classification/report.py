import pandas as pd
from sklearn.metrics import f1_score
import os, argparse
import glob
import re
import xlsxwriter

parser = argparse.ArgumentParser(description='Generate reports based on classification results.')
parser.add_argument('--outputdatasets', dest='outputdatasets', default="results/",help='dataset folder where the results are saved', type=str)
parser.add_argument('--methods',dest='methods',nargs='+',default=["COT","HC_Data","HC_FullDist","HC_SepState","HC_FullDist_Pseudocount","HC_SepState_Pseudocount","NB_Data","NB_FullDist","NB_SepState","NB_FullDist_Pseudocount","NB_SepState_Pseudocount","Random"], help="Specify one or more training methods.")
parser.add_argument('--model',dest='model',default="gpt-4o", help="Specify one or more models.")
parser.add_argument('--splits',dest='splits',nargs='+',default=[-1,10,20], help="Specify one or more splits.")
parser.add_argument('--output_prefix', dest='output_prefix', default="", help='Prefix for output report files', type=str)
args = parser.parse_args()

args.models = [args.model, "Nollm"]
args.outputdatasets = os.path.join(os.getcwd(), args.outputdatasets)


def load_result_file(method, model, split):
    pattern = f"method_{method}_llm_{model}_split_{split}_run_*.csv"
    file_paths = glob.glob(os.path.join(args.outputdatasets, pattern))
    
    if not file_paths:
        print(f"Warning: No files found matching pattern: {pattern}")
        return None

    all_runs_dfs = []
    for file_path in file_paths:
        match = re.search(r"run_(\d+)\.csv$", file_path)
        if match:
            run_id = match.group(1)
            df = pd.read_csv(file_path)
            df['run_id'] = run_id
            all_runs_dfs.append(df)
        else:
            print(f"Warning: Could not extract run_id from {file_path}")
    
    if not all_runs_dfs:
        return None

    combined_df = pd.concat(all_runs_dfs, ignore_index=True)
    model_name = f"{method}_{model}_split_{split}"
    combined_df['model'] = model_name
    return combined_df

dataframes = []
if "COT" in args.methods:
    pattern = f"method_COT_llm_{args.model}_run_*.csv"
    cot_files = glob.glob(os.path.join(args.outputdatasets, pattern))
    
    if cot_files:
        all_cot_dfs = []
        for cot_file in cot_files:
            match = re.search(r"run_(\d+)\.csv$", cot_file)
            if match:
                run_id = match.group(1)
                cot_df = pd.read_csv(cot_file)
                cot_df['run_id'] = run_id
                all_cot_dfs.append(cot_df)
            else:
                print(f"Warning: Could not extract run_id from {cot_file}")
        
        if all_cot_dfs:
            combined_cot_df = pd.concat(all_cot_dfs, ignore_index=True)
            if "pred_no_cot" in combined_cot_df.columns and "pred_cot" in combined_cot_df.columns:
                
                # COT
                cot_df = combined_cot_df.copy()
                cot_df = cot_df.rename(columns={"pred_cot": "predicted"}).drop(columns=["pred_no_cot"])
                cot_df['model'] = "COT"
                dataframes.append(cot_df)
            else:
                combined_cot_df['model'] = "COT"
                dataframes.append(combined_cot_df)
    else:
        print(f"Warning: No COT files found matching pattern: {pattern}")


for method in args.methods:
    if method == "COT":
        continue
        
    for model in args.models:
        if "Data" in method and model != "Nollm":
            continue
        for split in args.splits:
            df = load_result_file(method, model, split)
            if df is not None:
                dataframes.append(df)

if dataframes:
    df_merged = pd.concat(dataframes, axis=0, ignore_index=True)
else:
    print("Error: No result files found matching the specified criteria.")
    exit(1)

def rename_model(model_name):
    if "random" in model_name.lower():
        return "Random"
    elif model_name == "Direct_QA":
        return "QA"
    elif "HC_Data_Nollm_" in model_name:
        return model_name.replace("HC_Data_Nollm_", "HC Data")
    elif "HC_SepState_Pseudocount_" in model_name:
        return model_name.replace("HC_SepState_Pseudocount_", "HC EDP SepState")
    elif "HC_SepState_" in model_name:
        return model_name.replace("HC_SepState_", "HC SepState")
    elif "HC_FullDist_Pseudocount_" in model_name:
        return model_name.replace("HC_FullDist_Pseudocount_", "HC EDP FullDist")
    elif "HC_FullDist_" in model_name:
        return model_name.replace("HC_FullDist_", "HC FullDist")
    elif "NB_Data_Nollm_" in model_name:
        return model_name.replace("NB_Data_Nollm_", "NB Data")
    elif "NB_SepState_Pseudocount_" in model_name:
        return model_name.replace("NB_SepState_Pseudocount_", "NB EDP SepState")
    elif "NB_SepState_" in model_name:
        return model_name.replace("NB_SepState_", "NB SepState")
    elif "NB_FullDist_Pseudocount_" in model_name:
        return model_name.replace("NB_FullDist_Pseudocount_", "NB EDP FullDist")
    elif "NB_FullDist_" in model_name:
        return model_name.replace("NB_FullDist_", "NB FullDist")
    else:
        return model_name

has_run_id = 'run_id' in df_merged.columns

if has_run_id:
    print("\nDetected multiple runs. Will calculate metrics for each run and then average them.")

    run_metrics = []
    for (dataset, model, run_id), group in df_merged.groupby(['dataset', 'model', 'run_id']):
        y_true = group['actual']
        y_pred = group['predicted']
        split = model.split("_split_")[-1] if "_split_" in model else "-1"
        dataset_short = dataset.split('_')[0]
        
        run_metrics.append({
            "dataset": dataset_short,
            "model": model,
            "split": split,
            "run_id": run_id,
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0)
        })

    run_metrics_df = pd.DataFrame(run_metrics)
    results = run_metrics_df.groupby(['dataset', 'model', 'split']).agg({
        'f1_weighted': ['mean', 'std']
    }).reset_index()
    results.columns = ['_'.join(col).strip('_') for col in results.columns.values]
    results = results.rename(columns={
        'f1_weighted_mean': 'f1_weighted',
        'f1_weighted_std': 'f1_weighted_std'
    })
    print("\nAveraging metrics across multiple runs:")
    for (dataset, model, split), group in run_metrics_df.groupby(['dataset', 'model', 'split']):
        run_count = len(group)
        if run_count > 1:
            avg_f1 = group["f1_weighted"].mean()
            std_f1 = group["f1_weighted"].std()
            print(f"  {dataset}, {model}, split {split}: {run_count} runs, avg f1: {avg_f1:.2f}±{std_f1:.2f}")
    results['model'] = results['model'].apply(rename_model)
    skip_evaluation = True
else:
    df_wide = (
        df_merged
          .pivot(index=['dataset', 'row_id', 'actual'],
                 columns='model',
                 values='predicted')
          .reset_index()
    )
    merged_predictions_file = f"{args.output_prefix}merged_predictions.csv" if args.output_prefix else "merged_predictions.csv"
    df_wide.to_csv(os.path.join(args.outputdatasets, merged_predictions_file), index=False)
    skip_evaluation = False

def add_majority_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Add ensemble predictions via row-wise majority vote (ties → first model)."""
    df = df.copy()
    
    # Get all model columns (excluding dataset, row_id, actual)
    model_columns = [col for col in df.columns if col not in ['dataset', 'row_id', 'actual']]
    
    # Group models by method type
    nb_models = [col for col in model_columns if col.startswith('NB_')]
    hc_models = [col for col in model_columns if col.startswith('HC_')]
    
    # Add majority votes if we have enough models
    if len(nb_models) >= 2:
        df["NB_Majority"] = df[nb_models].mode(axis=1)[0]
    
    if len(hc_models) >= 2:
        df["HC_Majority"] = df[hc_models].mode(axis=1)[0]
    
    # If we have both Direct_QA and COT, add them to majority votes
    if "Direct_QA" in model_columns and "COT" in model_columns:
        if len(nb_models) >= 1:
            df["NB_QA_Majority"] = df[nb_models + ["Direct_QA", "COT"]].mode(axis=1)[0]
        
        if len(hc_models) >= 1:
            df["HC_QA_Majority"] = df[hc_models + ["Direct_QA", "COT"]].mode(axis=1)[0]
    
    return df


def evaluate_by_dataset(
    df: pd.DataFrame,
    label_col: str,
    pred_cols: list[str],
) -> pd.DataFrame:
    # Check if run_id column exists in the dataframe
    has_run_id = 'run_id' in df.columns
    
    records = []
    
    # Group by dataset and run_id if available
    if has_run_id:
        # First calculate metrics for each dataset, model, and run_id combination
        run_records = []
        for (dataset, run_id), g in df.groupby(["dataset", "run_id"]):
            # Use only the first part of the dataset name before underscore
            dataset_short = dataset.split('_')[0]
            y_true = g[label_col]

            for col in pred_cols:
                # Skip columns that don't exist in the dataframe
                if col not in g.columns:
                    print(f"Warning: Column '{col}' not found in the dataframe for dataset '{dataset}', run_id '{run_id}'")
                    continue
                    
                y_pred = g[col]
                run_records.append(
                    {
                        "dataset": dataset_short,
                        "model": col,
                        "split": col.split("_split_")[-1] if "_split_" in col else "-1",  # Extract split from model name
                        "run_id": run_id,
                        "f1_weighted": f1_score(
                            y_true, y_pred, average="weighted", zero_division=0
                        ),
                    }
                )
        
        # Convert to DataFrame
        run_df = pd.DataFrame(run_records)
        
        # Now calculate the average metrics and standard deviation across runs for each dataset and model
        if not run_df.empty:
            # Group by dataset, model, and split, then calculate mean and std of metrics
            avg_df = run_df.groupby(["dataset", "model", "split"]).agg({
                "f1_weighted": ["mean", "std"]
            })
            
            # Flatten the multi-level columns
            avg_df.columns = ['_'.join(col).strip('_') for col in avg_df.columns.values]
            
            # Rename columns for clarity
            avg_df = avg_df.rename(columns={
                'f1_weighted_mean': 'f1_weighted',
                'f1_weighted_std': 'f1_weighted_std'
            }).reset_index()
            
            # Add to records
            records = avg_df.to_dict('records')
            
            # Print information about averaging
            print("\nAveraging metrics across multiple runs:")
            for (dataset, model, split), group in run_df.groupby(["dataset", "model", "split"]):
                run_count = len(group)
                if run_count > 1:
                    avg_f1 = group["f1_weighted"].mean()
                    std_f1 = group["f1_weighted"].std()
                    print(f"  {dataset}, {model}, split {split}: {run_count} runs, avg f1: {avg_f1:.2f}±{std_f1:.2f}")
    else:
        # Original behavior if no run_id column
        for dataset, g in df.groupby("dataset"):
            # Use only the first part of the dataset name before underscore
            dataset_short = dataset.split('_')[0]
            y_true = g[label_col]

            for col in pred_cols:
                # Skip columns that don't exist in the dataframe
                if col not in g.columns:
                    print(f"Warning: Column '{col}' not found in the dataframe for dataset '{dataset}'")
                    continue
                    
                y_pred = g[col]
                records.append(
                    {
                        "dataset": dataset_short,
                        "model": col,
                        "split": col.split("_split_")[-1] if "_split_" in col else "-1",  # Extract split from model name
                        "f1_weighted": f1_score(
                            y_true, y_pred, average="weighted", zero_division=0
                        ),
                    }
                )

    return pd.DataFrame(records)

# If we haven't already calculated metrics for multiple runs
if not has_run_id or not 'skip_evaluation' in locals() or not skip_evaluation:
    # Print available columns for debugging
    print("\nAvailable columns in the dataframe:")
    print(df_wide.columns.tolist())
    
    # Skip adding majority votes as per requirement
    df = df_wide
    
    # Get all model columns (excluding dataset, row_id, actual)
    model_cols = [col for col in df.columns if col not in ['dataset', 'row_id', 'actual']]
    print(f"\nEvaluating {len(model_cols)} models: {model_cols}")
    
    # Evaluate models
    results = evaluate_by_dataset(df, "actual", model_cols)

# Apply renaming to model column (if not already done for multiple runs)
if not has_run_id or not 'results' in locals():
    results['model'] = results['model'].apply(rename_model)

# --- create reports for each split (dataset × model) --------------------------
# Create Excel writer
excel_file = f"{args.output_prefix}reports.xlsx" if args.output_prefix else "reports.xlsx"
excel_path = os.path.join(args.outputdatasets, excel_file)
writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

# Get unique splits
splits = results['split'].unique()

for split in splits:
    # Filter results for this split
    split_results = results[results['split'] == split]
    
    # Check if we have standard deviation columns
    has_std = 'f1_weighted_std' in split_results.columns
    
    # Create F1 report for this split
    f1_report = split_results.pivot(index="dataset", columns="model", values="f1_weighted")
    
    # Add average row
    f1_report.loc['Average'] = f1_report.mean()
    
    # If we have standard deviation data, create formatted reports with ±
    if has_std:
        # Create std report
        f1_std_report = split_results.pivot(index="dataset", columns="model", values="f1_weighted_std")
        
        # Add average row for std (using mean of std values)
        f1_std_report.loc['Average'] = f1_std_report.mean()
        
        # Create formatted report with ± for Excel
        f1_formatted = f1_report.round(2).astype(str)
        
        # Add ± and std deviation where std exists
        for col in f1_report.columns:
            for idx in f1_report.index:
                if not pd.isna(f1_std_report.loc[idx, col]):
                    f1_formatted.loc[idx, col] = f"{f1_report.loc[idx, col]:.2f}±{f1_std_report.loc[idx, col]:.2f}"
        
        # Save formatted report to Excel
        f1_formatted.to_excel(writer, sheet_name=f'F1_split_{split}')
        
        # Print formatted report
        print(f"\nWeighted-F1 per dataset & model for split {split}")
        print(f1_formatted.to_string())
    else:
        # Round to 2 decimal places before saving to Excel
        f1_report_rounded = f1_report.round(2)
        
        # Save to Excel
        f1_report_rounded.to_excel(writer, sheet_name=f'F1_split_{split}')
        
        # Print report
        print(f"\nWeighted-F1 per dataset & model for split {split}")
        print(f1_report.round(2).to_string())

# Save Excel file
writer.close()
print(f"\nReports saved to {excel_path}")

