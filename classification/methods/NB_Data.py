from typing import Dict, List, Tuple
import pandas as pd, numpy as np, os
from tqdm.auto import tqdm
from classification.utils.utils import create_naive_bayes_network, infer_target, epsilon, _f1
from PreprocessingBayesianNetworks.save_model_pickles import safe_save
from classification.utils.datasets import iter_train_test


def run_NB_Data(result_file_name,data_file_folder,split,run_id):

    def learn_parameters_naive_bayes_mle(
        df: pd.DataFrame,
        target_col: str,
        features: List[str],
        target_values: List[str],
        feature_values: Dict[str, List[str]]
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:

        target_counts = df[target_col].value_counts()
        total_samples = len(df)

        priors = {}
        for val in target_values:
            count = target_counts.get(val, 0)
            # Apply Laplace smoothing (add-one) to avoid zero probabilities
            priors[str(val)] = (count + 1) / (total_samples + len(target_values))

        # Ensure priors sum to 1
        prior_sum = sum(priors.values())
        if abs(prior_sum - 1.0) > epsilon:  # Allow for small floating-point errors
            print(f"Warning: Prior probabilities sum to {prior_sum}, normalizing...")
            for val in priors:
                priors[val] = priors[val] / prior_sum

        cpts = {}
        for feature in features:
            feature_vals = feature_values[feature]
            cpt_array = np.zeros((len(feature_vals), len(target_values)))

            for i, target_val in enumerate(target_values):
                target_df = df[df[target_col] == target_val]
                target_count = len(target_df)

                for j, feature_val in enumerate(feature_vals):
                    count = len(target_df[target_df[feature] == feature_val])
                    cpt_array[j, i] = (count + 1) / (target_count + len(feature_vals))

                col_sum = cpt_array[:, i].sum()
                if abs(col_sum - 1.0) > epsilon:
                    print(f"Warning: CPT column for feature '{feature}' and target '{target_val}' sums to {col_sum}, normalizing...")
                    cpt_array[:, i] = cpt_array[:, i] / col_sum
            cpts[feature] = cpt_array
        return priors, cpts

    records = []

    for name, tr, td, te, target, desc, features, feature_values, target_values, df_merged in iter_train_test(split, 0.8,run_id=run_id):
        print(f"Processing dataset: {name}")

        priors, cpts = learn_parameters_naive_bayes_mle(
            df=tr,
            target_col=target,
            features=features,
            target_values=target_values,
            feature_values=feature_values
        )

        print(f"Prior probabilities: {priors}")

        model = create_naive_bayes_network(
            target_col=target,
            features=features,
            target_values=target_values,
            feature_values=feature_values,
            cpts=cpts,
            priors=priors
        )

        model_path = os.path.join(data_file_folder, f"{name}.pkl")
        safe_save(model, model_path)
        print(f"Model saved to {model_path}")

        for idx, row in tqdm(te.iterrows(), total=len(te), desc=f"Evaluating {name}"):
            evidence = {}
            for feature in features:
                evidence[feature] = str(row[feature])

            predicted = infer_target(model, evidence, target)
            records.append({
                "dataset": name,
                "row_id": idx,
                "actual": row[target],
                "predicted": predicted,
            })

    # ─────────────────────────── persist + metrics ───────────────────────── #
    pred_df = pd.DataFrame.from_records(records)
    pred_df.to_csv(result_file_name, index=False)

    summary = (
        pred_df
        .groupby("dataset")
        .apply(lambda g: pd.Series({
            "n_rows": len(g),
            "f1": _f1(g["actual"], g["predicted"]),
        }))
        .reset_index()
    )

    print("\n✔ Evaluation complete. Metric snapshot:\n", summary)
