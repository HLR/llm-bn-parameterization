import csv, pandas as pd, random
from typing import Dict, Any, List, Tuple
import concurrent.futures
from tqdm.auto import tqdm
from classification.utils.datasets import iter_train_test, get_dataset_descriptions
from classification.utils.utils import _f1

def run_Random_method(result_file_name,NUM_WORKERS=8):

    def process_row(args: Tuple[str, str, str, List[Any], pd.Series, int]) -> Dict[str, Any]:
        name, desc, target, target_values, row, idx = args

        return {
            "dataset": name,
            "row_id": idx,
            "actual": row[target],
            "predicted": random.choice(target_values),
        }

    records: List[Dict[str, Any]] = []

    for name, tr, td, te, target, desc, features, feature_values, target_values, df_merged in iter_train_test(-1,0.8):

        desc = get_dataset_descriptions(name)
        target_values = sorted(te[target].dropna().unique())
        tasks = [(name, desc, target, target_values, row, idx) for idx, row in te.iterrows()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(tqdm(
                executor.map(process_row, tasks),
                total=len(tasks),
                desc=f"Processing {name}",
                leave=False
            ))
        records.extend(results)

    pred_df = pd.DataFrame.from_records(records)
    pred_df.to_csv(result_file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

    pred_df = pd.read_csv(result_file_name)

    summary = (
        pred_df.groupby("dataset")
          .apply(lambda g: pd.Series({
              "n_rows":      len(g),
              "f1":   _f1(g["actual"], g["predicted"]),

          }))
          .reset_index()
    )

    print("\n✔ Evaluation complete. Metric snapshot:\n", summary)
