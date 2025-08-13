import csv, time, pandas as pd
from typing import Dict, Any, List, Tuple
import concurrent.futures

from tqdm.auto import tqdm
from classification.utils.datasets import iter_train_test, get_dataset_descriptions
from classification.utils.prompts import build_labeling_prompt
from classification.utils.utils import _f1


def run_COT_method(chat_model,result_file_name, NUM_WORKERS = 4):

    def call_llm(prompt: str,is_COT: bool, target_values: list) -> str:
        while True:
            try:
                llm_response = chat_model.invoke(prompt).content.lower()
                if is_COT:
                    if "answer:" in llm_response:
                        llm_response = llm_response.split("answer:")[-1].split("*")[0].strip(" *[ '\"]")
                    else:
                        llm_response = llm_response.splitlines()[-1].strip()
                    answer = llm_response.split("*")[0].strip(" *[ '\"]")
                else:
                    answer = llm_response.split("*")[0].strip(" *[ '\"]")
                assert answer in target_values, f"Invalid answer: {answer} targets: {target_values}"
                return answer
            except Exception as e:
                print(f"Error calling LLM: {e}")
                print(prompt, is_COT, target_values)
                time.sleep(1)

    def process_row(args: Tuple[str, str, str, List[Any], pd.Series, int]) -> Dict[str, Any]:
        name, desc, target, target_values, row, idx = args

        prompt_cot = build_labeling_prompt(name=name, description=desc, target_col=target, target_values=target_values, row=row, with_cot=True)
        pred_cot = call_llm(prompt=prompt_cot,is_COT=True,target_values=target_values)

        return {
            "dataset": name,
            "row_id": idx,
            "actual": row[target],
            "pred_cot": pred_cot,
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
              "f1_cot":      _f1(g["actual"], g["pred_cot"]),
          }))
          .reset_index()
    )

    print("\n✔ Evaluation complete. Metric snapshot:\n", summary)
