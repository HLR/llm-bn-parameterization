import os, sys, pandas as pd
from tqdm.auto import tqdm
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PreprocessingBayesianNetworks.save_model_pickles import safe_save
from classification.utils.datasets import iter_train_test
from classification.utils.utils import _f1, infer_target


def run_HC_Data(result_file_name,data_file_folder,split,run_id):

    def learn_structure_hc(df_merged: pd.DataFrame, state_names) -> BayesianNetwork:
        hc = HillClimbSearch(data = df_merged,state_names=state_names)
        best_model = hc.estimate(scoring_method = BicScore(df_merged))
        return BayesianNetwork(best_model)

    records = []
    for name, tr, td, te, target, _desc, features, _feature_values, _target_values, df_merged in iter_train_test(split,0.80,run_id=run_id):
        print(f"\n▶ Processing {name}")
        bn = learn_structure_hc(df_merged,_feature_values | {target:_target_values})
        bn.fit(data = tr,state_names = _feature_values | {target:_target_values})
        model_path = os.path.join(data_file_folder, f"{name}.pkl")
        safe_save(bn, model_path)
        print(f"   ↳ model saved → {model_path}")

        for idx, row in tqdm(te.iterrows(), total=len(te), desc=f"Evaluating {name}"):
            evidence = {}
            for feature in features:
                evidence[feature] = str(row[feature])

            predicted = infer_target(bn, evidence, target)
            records.append({
                "dataset": name,
                "row_id": idx,
                "actual": row[target],
                "predicted": predicted,
            })

    pred_df = pd.DataFrame.from_records(records)
    pred_df.to_csv(result_file_name, index=False)

    summary = (
        pred_df.groupby("dataset")
        .apply(
            lambda g: pd.Series(
                {
                    "n_rows": len(g),
                    "f1": _f1(g["actual"], g["predicted"]),
                }
            )
        )
        .reset_index()
    )

    print("\n✔ Evaluation complete. Metric snapshot:\n", summary)