import os, pandas as pd, numpy as np
from tqdm.auto import tqdm
from classification.utils.datasets import iter_train_test
from PreprocessingBayesianNetworks.save_model_pickles import safe_save,safe_load
from classification.utils.utils import create_naive_bayes_network, infer_target, epsilon, _f1
from classification.utils.prompts import build_NB_SepState_prior_prompt, build_NB_SepState_dependant_prompt


def run_NB_SepState(chat_model, result_file_name, data_file_folder):

    records = []
    
    for name, tr, td, te, target, desc, features, feature_values, target_values, df_merged in iter_train_test(-1,0.8):
    
        print(f"Processing dataset: {name}")
        model_path = os.path.join(data_file_folder, f"{name}.pkl")
        if os.path.exists(model_path):
            model = safe_load(model_path)
        else:
            # Query LLM for prior probabilities one by one
            prior_prompt, prior_fixing_parser = build_NB_SepState_prior_prompt(chat_model)
            unnormalized_priors = {}
            
            for t_val in target_values:
                prior_messages = prior_prompt.format_messages(
                    dataset_name=name, 
                    dataset_desc=desc, 
                    target_var=target, 
                    target_value=t_val
                )
                prior_response = chat_model.invoke(prior_messages)
                prior_result = prior_fixing_parser.invoke(prior_response.content)
                unnormalized_priors[t_val.lower()] = prior_result.probability
            
            # Normalize prior probabilities
            total_prior = sum(unnormalized_priors.values())
            if total_prior <= 0:
                raise ValueError(f"All prior probabilities are zero or negative")
            
            priors = {k: v / total_prior for k, v in unnormalized_priors.items()}
            print(f"Prior probabilities: {priors}")

            # Query LLM for conditional probabilities one by one
            column_prompt, single_fix_parser = build_NB_SepState_dependant_prompt(chat_model)
            cpts = {}
            
            for feature in tqdm(features, desc="Querying LLM for CPTs"):
                col_distributions = {}
    
                for t_val in target_values:
                    unnormalized_dist = {}
                    
                    for f_val in feature_values[feature]:
                        msgs = column_prompt.format_messages(
                            dataset_name=name,
                            dataset_desc=desc,
                            target_var=target,
                            target_value=t_val,
                            feature=feature,
                            feature_value=f_val
                        )
    
                        raw = chat_model.invoke(msgs)
                        prob = single_fix_parser.invoke(raw.content).probability
                        unnormalized_dist[f_val.lower()] = prob
                    
                    # Normalize the probabilities
                    total = sum(unnormalized_dist.values())
                    if total <= 0:
                        raise ValueError(f"{feature}|{t_val}: all probabilities are zero or negative")
                    
                    dist = {k: v / total for k, v in unnormalized_dist.items()}
                    col_distributions[t_val] = dist
    
                for t_val, dist in col_distributions.items():
                    total = sum(dist.values())
                    if not np.isclose(total, 1, atol=epsilon):
                        raise ValueError(f"{feature}|{t_val}: probs sum to {total:.3f}, not 1.")
    
                col_order = [col_distributions[t][fv.lower()] for fv in feature_values[feature] for t in target_values]
                cpt_array = np.array(col_order).reshape(
                    len(feature_values[feature]),
                    len(target_values)
                )
    
                cpts[feature] = cpt_array
    
            model = create_naive_bayes_network(
                target_col=target,
                features=features,
                target_values=target_values,
                feature_values=feature_values,
                cpts=cpts,
                priors=priors
            )
            safe_save(model, model_path)
            print(f"Created and saved new model to {model_path}")
    
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