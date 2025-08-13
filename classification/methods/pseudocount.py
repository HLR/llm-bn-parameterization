import os, sys, pandas as pd
from tqdm.auto import tqdm
from pgmpy.models import BayesianNetwork
from classification.utils.datasets import iter_train_test
from classification.utils.utils import load_naive_bayes_models, _f1, infer_target

def run_pseudocount(BN_LLM_DIR,result_file_name,split,run_id):

    print("Loading Bayesian networks from the Naive Bayes Folder...")
    models = load_naive_bayes_models(folder_name=BN_LLM_DIR)
    print("Bayesian networks loaded.")
    records = []
    for name, tr, td, te, target, desc, features, feature_values, target_values, df_merged in tqdm(list(iter_train_test(split,0.80,run_id=run_id)), desc="Processing datasets"):
        print(f"\nProcessing dataset: {name}")
        model = models[name]
        sample_size = len(tr)

        ess_values = [("half", sample_size // 2),("equal", sample_size),("twice", sample_size * 2)]
        ess_index = 0
        if td is not None:
            dev_acc = []
            for ess_name, ess_value in ess_values:
                print(f"Updating model with ESS = {ess_value} ({ess_name} of sample size)")
                dev_records = []
                updated_model = BayesianNetwork(model.edges())
                updated_model.add_nodes_from(model.nodes())
                for cpd in model.get_cpds():
                    updated_model.add_cpds(cpd)

                updated_model.fit_update(tr, n_prev_samples=ess_value)

                for idx, row in tqdm(td.iterrows(), total=len(td), desc=f"Evaluating Dev {name}"):
                    evidence = {}
                    for feature in features:
                        evidence[feature] = str(row[feature])

                    predicted = infer_target(updated_model, evidence, target)
                    dev_records.append({
                        "dataset": name,
                        "row_id": idx,
                        "ess_type": ess_name,
                        "actual": row[target],
                        "predicted": predicted,
                    })
                dev_acc.append(_f1(pd.DataFrame.from_records(dev_records)["actual"], pd.DataFrame.from_records(dev_records)["predicted"]))
            ess_index=dev_acc.index(max(dev_acc))

        updated_model = BayesianNetwork(model.edges())
        updated_model.add_nodes_from(model.nodes())
        for cpd in model.get_cpds():
            updated_model.add_cpds(cpd)
        updated_model.fit_update(tr, n_prev_samples=ess_values[ess_index][1])

        for idx, row in tqdm(te.iterrows(), total=len(te), desc=f"Evaluating Dev {name}"):
            evidence = {}
            for feature in features:
                evidence[feature] = str(row[feature])

            predicted = infer_target(updated_model, evidence, target)
            records.append({
                "dataset": name,
                "row_id": idx,
                "ess_type": ess_values[ess_index][0],
                "actual": row[target],
                "predicted": predicted,
            })


    pred_df = pd.DataFrame.from_records(records)
    pred_df.to_csv(result_file_name, index=False)

    summary = (
        pred_df
        .groupby("dataset")
        .apply(lambda g: pd.Series({
            "n_rows":   len(g),
            "f1":       _f1(g["actual"], g["predicted"]),
        }))
        .reset_index()
    )

    print("\n✔ Evaluation complete. Metric snapshot:\n", summary)

