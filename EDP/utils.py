import pandas as pd
from sklearn.metrics import r2_score
from pgmpy.readwrite import BIFWriter
from pgmpy.factors.discrete import TabularCPD
import os
from collections import defaultdict
import itertools
import numpy as np
import math
import seaborn as sns

def read_names_from_csv(csv_path):
    """Read list of names from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Assuming the column with names is called 'name'
        return df['name'].tolist(),df['explanation_dict'].tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def create_text(model, grouped=False):
    """
    Takes in a pgmpy model with a query including (query_nodes, evidence_nodes), and the values of these variables in the query.
    If grouped=True, returns a dictionary grouping states by their parent conditions and a list of possible states for each node.

    Args:
        model: A pgmpy BayesianNetwork model
        grouped: Boolean indicating whether to return grouped format (default: False)

    Returns:
        If grouped=False: String with textual explanation
        If grouped=True: Tuple of (dict of grouped states, dict of possible states)
    """
    context = ""
    grouped_states = {}  # Dictionary to store grouped states
    node_possible_states = {}  # Dictionary to store possible states for each node

    cpds = model.get_cpds()
    for cpd in cpds:
        variable = cpd.variable
        parents = model.get_parents(variable)
        state_names = cpd.state_names
        variable_states = state_names[variable]
        parent_states = {parent: state_names[parent] for parent in parents}

        # Store possible states for each node
        node_possible_states[variable] = variable_states

        if not parents:
            # Variable has no parents, prior probability
            if grouped:
                grouped_states[f"{variable} (prior)"] = []

            for i, prob in enumerate(cpd.values):
                state = variable_states[i]
                prob_percent = prob * 100
                text = f"{variable} is {state} with probability of {prob_percent:.2f}%."

                if grouped:
                    grouped_states[f"{variable} (prior)"].append(text)
                else:
                    context += text + "\n"
        else:
            # Variable has parents, conditional probabilities
            import itertools
            parent_state_names = [state_names[parent] for parent in parents]
            parent_state_combinations = list(itertools.product(*parent_state_names))
            variable_state_names = state_names[variable]

            # Get the indices for the variable states
            var_indices = range(len(variable_state_names))
            for parent_state_combo in parent_state_combinations:
                parent_state_dict = dict(zip(parents, parent_state_combo))
                parent_indices = [state_names[parent].index(state) for parent, state in parent_state_dict.items()]

                # Create key for grouped states
                conditions = [f"{parent} is {state}" for parent, state in parent_state_dict.items()]
                conditions_text = " and ".join(conditions)
                group_key = f"{variable} when {conditions_text}"

                if grouped:
                    grouped_states[group_key] = []

                for var_index in var_indices:
                    var_state = variable_state_names[var_index]
                    indices = (var_index,) + tuple(parent_indices)
                    prob = cpd.values[indices]
                    prob_percent = prob * 100

                    text = f"If {conditions_text}, then {variable} is {var_state} with probability of {prob_percent:.2f}%."

                    if grouped:
                        grouped_states[group_key].append(text)
                    else:
                        context += text + "\n"

    if grouped:
        return grouped_states, node_possible_states
    return context

def kl_divergence(p, q, eps=1e-8):
    """
    Computes the KL divergence D_KL(P || Q) for discrete distributions p, q.
    Both p and q should be arrays that sum to 1.
    We add a small epsilon to avoid log(0).
    """
    p = np.array(p, dtype=np.float64)+ eps
    q = np.array(q, dtype=np.float64)+ eps

    return np.sum(p * np.log(np.maximum(p, eps) / np.maximum(q, eps)))

def save_model_as_bif(model, cpt_dict, output_dir, suffix,just_return_it=False):

    new_model = model.copy()

    # Update CPTs for each node
    for node in new_model.nodes():
        old_cpd = model.get_cpds(node)
        parents = list(model.get_parents(node))
        node_states = list(old_cpd.state_names[node])
        if not parents:
            probabilities = [cpt_dict[node].get(state, 0.0) for state in node_states]
            prob_array = np.array(probabilities).reshape(-1, 1)
            new_cpd = TabularCPD(
                variable=node,
                variable_card=len(node_states),
                values=prob_array,
                state_names={node: node_states}
            )
        else:
            parent_states = [old_cpd.state_names[parent] for parent in parents]
            parent_cards = [len(states) for states in parent_states]
            from itertools import product
            parent_combinations = list(product(*parent_states))
            prob_array = np.zeros((len(node_states), np.prod(parent_cards)))
            for col_idx, parent_comb in enumerate(parent_combinations):
                for row_idx, state in enumerate(node_states):
                    prob = cpt_dict[node][parent_comb].get(state, 0.0)
                    prob_array[row_idx, col_idx] = prob
            new_cpd = TabularCPD(
                variable=node,
                variable_card=len(node_states),
                values=prob_array,
                evidence=parents,
                evidence_card=parent_cards,
                state_names={node: node_states, **{p: s for p, s in zip(parents, parent_states)}}
            )
        new_model.remove_cpds(old_cpd)
        new_model.add_cpds(new_cpd)

    if just_return_it:
        return new_model
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model.name}_{suffix}.bif")
    writer = BIFWriter(new_model)
    writer.write_bif(output_path)
    return output_path

def save_all_model_versions(model, initial_cpts, learned_cpts, combined_cpts,sample_num,prior_weight, llm_model, sampling_method, output_dir,create_data_baselines=False):
    """
    Save all three versions of a Bayesian Network model.

    Args:
        model: Original pgmpy BayesianNetwork model
        initial_cpts: Dictionary of initial CPTs
        learned_cpts: Dictionary of CPTs learned from data
        combined_cpts: Dictionary of combined CPTs
        output_dir: Directory to save the BIF files
    """
    # Create model-specific output directory
    if create_data_baselines:
        model_output_dir = os.path.join(output_dir, f"{sampling_method}/{sample_num}/{model.name}")
    else:
        model_output_dir = os.path.join(output_dir, f"{model.name}_{prior_weight}_{sample_num}_{llm_model}_{sampling_method}")
    os.makedirs(model_output_dir, exist_ok=True)

    # Save original model
    if not create_data_baselines:
        original_path = os.path.join(model_output_dir, f"{model.name}_original.bif")
        writer = BIFWriter(model)
        writer.write_bif(original_path)

        initial_cpts_path = save_model_as_bif(model, initial_cpts, model_output_dir, "initial")
        learned_path = save_model_as_bif(model, learned_cpts, model_output_dir, "learned")
        combined_path = save_model_as_bif(model, combined_cpts, model_output_dir, "combined")

        return original_path, initial_cpts_path, learned_path, combined_path
    learned_path = save_model_as_bif(model, learned_cpts, model_output_dir, "learned")
    return None, None , learned_path, None

def evaluate_cpts(true_model, learned_cpts):
    true_probs, learned_probs, kl_values, cpt_sizes, comparisons = [], [], [], [], defaultdict(list)
    predicted_model = save_model_as_bif(true_model, learned_cpts, "", "", just_return_it=True)
    from EstimationofPriorProbabilitiesbyLLMs.utils.test_utils import entire_BN_KL_divergence
    BN_KL_divergence = entire_BN_KL_divergence(true_model,predicted_model)
    for node in true_model.nodes():
        true_cpd, parents, node_states = true_model.get_cpds(node), true_model.get_parents(node), true_model.get_cpds(node).state_names[node]
        num_node_states = len(node_states)
        if not parents:
            p_dist, q_dist = [], []
            for state_idx, state in enumerate(node_states):
                true_prob = get_cpd_value(true_cpd, [], state_idx)
                learned_prob = learned_cpts[node].get(state, 0.0)

                true_probs.append(true_prob)
                learned_probs.append(learned_prob)
                p_dist.append(true_prob)
                q_dist.append(learned_prob)

                comparisons[node].append({
                    'state': state,
                    'true_prob': true_prob,
                    'learned_prob': learned_prob,
                    'diff': abs(true_prob - learned_prob)
                })

            kl_values.append(kl_divergence(p_dist, q_dist))
            cpt_sizes.append(num_node_states)
        else:
            parent_states = [true_cpd.state_names[parent] for parent in parents]
            node_cpt_size = num_node_states * np.prod([len(s) for s in parent_states])
            for parent_comb in itertools.product(*parent_states):
                parent_dict = dict(zip(parents, parent_comb))
                p_dist, q_dist = [], []
                evidence_indices = []
                for parent, value in zip(parents, parent_comb):
                    evidence_indices.append(true_cpd.state_names[parent].index(value))

                for state_idx, state in enumerate(node_states):
                    true_prob = get_cpd_value(true_cpd, evidence_indices, state_idx)
                    learned_prob = learned_cpts[node][parent_comb].get(state, 0.0)

                    true_probs.append(true_prob)
                    learned_probs.append(learned_prob)

                    p_dist.append(true_prob)
                    q_dist.append(learned_prob)

                    comparisons[node].append({
                        'state': state,
                        'parents': parent_dict,
                        'true_prob': true_prob,
                        'learned_prob': learned_prob,
                        'diff': abs(true_prob - learned_prob)
                    })

                kl = kl_divergence(p_dist, q_dist)
                kl_values.append(kl)
                cpt_sizes.append(num_node_states)

    metrics = {
        'r2': r2_score(true_probs, learned_probs),
        'mae': np.mean(np.abs(np.array(true_probs) - np.array(learned_probs))),
        'CA25': sum(abs(t - l) <= 0.25 for t, l in zip(true_probs, learned_probs)) / len(true_probs),
        'CA10': sum(abs(t - l) <= 0.10 for t, l in zip(true_probs, learned_probs)) / len(true_probs),
        'CA1': sum(abs(t - l) <= 0.01 for t, l in zip(true_probs, learned_probs)) / len(true_probs)
    }

    avg_kl = np.mean(kl_values)
    weighted_kl = np.sum(np.array(kl_values) * np.array(cpt_sizes)) / np.sum(cpt_sizes)
    metrics['kl_avg'] = avg_kl
    metrics['kl_weighted'] = weighted_kl
    metrics['BN_KL_divergence'] = BN_KL_divergence

    return metrics, comparisons

def get_cpd_value(cpd, evidence_list, state_idx):
    if len(evidence_list) == 0:
        return cpd.values[state_idx]
    indices = [state_idx] + evidence_list
    return cpd.values[tuple(indices)]

def calculate_results(model, samples, prior_weight_hyper_parameter, sampling_method, LLM_model, metrics_initial, metrics_learned, metrics_combined, learning_method):

    return {
        'model_name': model.name,
        'learning_method': learning_method,
        'sample_num': samples,
        'sampling_method': sampling_method,
        'llm_model': LLM_model,
        'prior_weight': prior_weight_hyper_parameter,
        'initial_r2': metrics_initial['r2'],
        'learned_r2': metrics_learned['r2'],
        'combined_r2': metrics_combined['r2'],
        'initial_CA1': metrics_initial['CA1'],
        'learned_CA1': metrics_learned['CA1'],
        'combined_CA1': metrics_combined['CA1'],
        'initial_CA10': metrics_initial['CA10'],
        'learned_CA10': metrics_learned['CA10'],
        'combined_CA10': metrics_combined['CA10'],
        'initial_CA25': metrics_initial['CA25'],
        'learned_CA25': metrics_learned['CA25'],
        'combined_CA25': metrics_combined['CA25'],

        'initial_kl_avg': metrics_initial['kl_avg'],
        'learned_kl_avg': metrics_learned['kl_avg'],
        'combined_kl_avg': metrics_combined['kl_avg'],
        'initial_kl_weighted': metrics_initial['kl_weighted'],
        'learned_kl_weighted': metrics_learned['kl_weighted'],
        'combined_kl_weighted': metrics_combined['kl_weighted'],

        'initial_BN_KL_divergence': metrics_initial['BN_KL_divergence'],
        'learned_BN_KL_divergence': metrics_learned['BN_KL_divergence'],
        'combined_BN_KL_divergence': metrics_combined['BN_KL_divergence'],
    }


def n_params_from_graph(model,independent_variables_only=False) -> int:
    total = 0
    for node in model.nodes():
        card_x  = model.get_cardinality(node)
        parent_cards = [model.get_cardinality(p) for p in model.get_parents(node)]
        total += (card_x-independent_variables_only) * math.prod(parent_cards or [1])
    return total

def create_custom_palette(sorted_model_samples):
    palette_dict = {}
    color_cycle = sns.color_palette('Set2')+sns.color_palette('Set2')+sns.color_palette('Set2')+sns.color_palette('Set2')
    color_index = 3
    for ms in sorted_model_samples:
        if ms.startswith("MLE"):
            palette_dict[ms] = color_cycle[0]
        elif ms.startswith("EDP"):
            palette_dict[ms] = color_cycle[1]
        elif ms.startswith("Uniform") and not ms == "Uniform":
            palette_dict[ms] = color_cycle[2]
        else:
            palette_dict[ms] = color_cycle[color_index]
            color_index+=1
    return palette_dict

model_name_translation = {"GPT-4o (SepState)":"gpt-4o", "GPT-4o (FullDist)":"gpt-4o-inoneprompt", "GPT-4o (Token Probability)":"gpt-4o-tokenprob", "GPT-4o (Without Context)":"gpt-4o-withoutcontext", "GPT-4o (Random)":"gpt-4o-random",
    "GPT-4o-mini (SepState)":"gpt-4o-mini", "GPT-4o-mini (FullDist)":"gpt-4o-mini-inoneprompt", "GPT-4o-mini (Token Probability)":"gpt-4o-mini-tokenprob", "GPT-4o-mini (Without Context)":"gpt-4o-mini-withoutcontext", "GPT-4o-mini (Random)":"gpt-4o-mini-random",
    "O3 (SepState)":"o3", "O3 (FullDist)":"o3-inoneprompt",
    "O3-mini (SepState)":"o3-mini", "O3-mini (FullDist)":"o3-mini-inoneprompt",
    "DeekSeek-R1 (SepState)":"deepseek-R1", "DeekSeek-R1 (FullDist)":"deepseek-R1-inoneprompt",
    "DeekSeek-V3 (SepState)":"deepseek", "DeekSeek-V3 (FullDist)":"deepseek-inoneprompt", "DeekSeek-V3 (Token Probability)":"deepseek-tokenprob", "DeekSeek-V3 (Without Context)":"deepseek-withoutcontext", "DeekSeek-V3 (Random)":"deepseek-random",
    "Uniform":"Mean","Random":"Random",
    "Gemini Pro (SepState)":"gemini-pro","Claude 3.5 (SepState)":"claude-3.5","Gemini Pro (FullDist)":"gemini-pro-inoneprompt","Claude 3.5 (FullDist)":"claude-3.5-inoneprompt"
}