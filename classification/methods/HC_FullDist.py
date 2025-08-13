import os, itertools, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple
from tqdm.auto import tqdm
from classification.utils.prompts import build_HC_fulldist_prompt
from PreprocessingBayesianNetworks.save_model_pickles import safe_save, safe_load
from classification.utils.utils import infer_target, epsilon, _f1
from classification.utils.datasets import iter_train_test
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def run_HC_FullDist(chat_model,BN_STRUCT_DIR,result_file_name,data_file_folder,force=False):

    row_prompt, row_fix_parser = build_HC_fulldist_prompt(chat_model)

    def enumerate_parent_combos(parents: List[str], value_map: Dict[str, List[str]]):
        """Yield ordered tuples representing every combination of parent values."""
        parent_values_lists = [value_map[p] for p in parents]
        for combo in itertools.product(*parent_values_lists):
            yield combo


    def build_cpd(child: str, parents: List[str], value_map: Dict[str, List[str]]) -> TabularCPD:
        """Ask the LLM for every row of the CPT and assemble a pgmpy TabularCPD."""
        child_values = value_map[child]
        evidence_card = [len(value_map[p]) for p in parents]
        n_rows = len(child_values)
        n_cols = np.prod(evidence_card) if parents else 1

        cpt = np.zeros((n_rows, n_cols))

        if parents:
            # Enumerate parent value combinations **in lexical order** for reproducibility
            for col_idx, combo in enumerate(enumerate_parent_combos(parents, value_map)):
                assignment_str = ", ".join(f"{p}={v}" for p, v in zip(parents, combo))
                msgs = row_prompt.format_messages(
                    dataset_name=curr_ds_name,
                    dataset_desc=curr_ds_desc,
                    child=child,
                    child_values=child_values,
                    parent_assignment=assignment_str,
                )
                raw = chat_model.invoke(msgs)
                dist = row_fix_parser.invoke(raw.content).probabilities
                dist = {k.lower(): v for k, v in dist.items()}
                total = sum(dist.values())
                if not np.isclose(total, 1, atol=epsilon):
                    raise ValueError(f"{child}|{assignment_str}: probs sum to {total:.3f}")
                for row_idx, child_val in enumerate(child_values):
                    cpt[row_idx, col_idx] = dist[child_val.lower()]
        else:
            # Root node – prompt once
            msgs = row_prompt.format_messages(
                dataset_name=curr_ds_name,
                dataset_desc=curr_ds_desc,
                child=child,
                child_values=child_values,
                parent_assignment="<no parents> (prior)"
            )
            raw = chat_model.invoke(msgs)
            dist = row_fix_parser.invoke(raw.content).probabilities
            dist = {k.lower(): v for k, v in dist.items()}
            total = sum(dist.values())
            if not np.isclose(total, 1, atol=epsilon):
                raise ValueError(f"{child}: prior probs sum to {total:.3f}")
            for row_idx, child_val in enumerate(child_values):
                cpt[row_idx, 0] = dist[child_val.lower()]

        return TabularCPD(variable=child, variable_card=len(child_values),
                          values=cpt,
                          evidence=parents or None,
                          evidence_card=evidence_card or None, state_names = {child: value_map[child]} | {p:value_map[p] for p in parents})

    records: List[Dict[str, Any]] = []
    for curr_ds_name, tr_df, td_df, te_df, target_col, curr_ds_desc, features, feature_values, target_values, df_merged in iter_train_test(-1,0.8):

        print(f"\n▶ Processing dataset: {curr_ds_name}")
        output_llm_path = os.path.join(data_file_folder, f"{curr_ds_name}.pkl")
        if os.path.exists(output_llm_path) and not force:
            model = safe_load(output_llm_path)
        else:
            model_path = os.path.join(BN_STRUCT_DIR, f"{curr_ds_name}.pkl")

            structure_model = safe_load(model_path)
            model = BayesianNetwork(structure_model.edges())
            model.add_nodes_from(structure_model.nodes())

            value_map = feature_values | {target_col: target_values}

            cpds: List[TabularCPD] = []
            for child in tqdm(model.nodes(), desc="Estimating CPTs"):
                parents = list(model.get_parents(child))
                cpd = build_cpd(child, parents, value_map)
                cpds.append(cpd)

            model.add_cpds(*cpds)
            assert model.check_model()

            safe_save(model, output_llm_path)
            print(f"  • BN with CPTs saved to: {output_llm_path}")

        for idx, row in tqdm(te_df.iterrows(), total=len(te_df), desc=f"Evaluating {curr_ds_name}"):
            evidence = {col: str(row[col]).lower() for col in features if not "unknown" == str(row[col]).lower()}
            predicted = infer_target(model, evidence, target_col)
            records.append({
                "dataset":  curr_ds_name,
                "row_id":   idx,
                "actual":   row[target_col],
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