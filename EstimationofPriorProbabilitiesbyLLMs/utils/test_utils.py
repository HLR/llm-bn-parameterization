import re, math, itertools, numpy as np
from sklearn.metrics import r2_score
from tabulate import tabulate
from EDP.learner_utils import load_initial_cpts
from EDP.utils import save_model_as_bif
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
epsilon = 1e-8

def calculate_entropy(probabilities):
    entropies = []
    for p in probabilities:
        if p == 0 or p == 1:
            entropies.append(0)
        else:
            entropies.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    return np.mean(entropies)

def calculate_kl_divergence(true_probs, pred_probs, groups, node_names, normalized=True):

    kl_divs = []
    start_idx = 0
    while start_idx < len(true_probs):
        current_group = groups[start_idx]
        end_idx = start_idx
        while end_idx < len(groups) and groups[end_idx] == current_group:
            end_idx += 1

        true_group = true_probs[start_idx:end_idx]
        pred_group = pred_probs[start_idx:end_idx]


        true_group = np.array(true_group) + epsilon
        pred_group = np.array(pred_group) + epsilon

        kl_div = np.sum(true_group * np.log(true_group / pred_group))
        if np.sum(0>(true_group / pred_group)):
            print("there is a negative number", true_group,pred_group )
        if kl_div<-0.001 and normalized:
            print("Error: KL Divergence is negative", kl_divs)
        kl_divs.append(kl_div)
        if np.mean(kl_divs)<-0.001 and normalized:
            print("Error: KL Divergence is negative", kl_divs)
        start_idx = end_idx
    return np.mean(kl_divs)


def Jensen_Shannon_divergence(true_probs, pred_probs, groups, node_names, normalized=True):
    epsilon = 1e-10
    js_divs = []

    start_idx = 0
    while start_idx < len(true_probs):
        current_group = groups[start_idx]
        end_idx = start_idx
        # collect all indices belonging to the same group
        while end_idx < len(groups) and groups[end_idx] == current_group:
            end_idx += 1

        # extract the probabilities for the current group
        true_group = np.array(true_probs[start_idx:end_idx]) + epsilon
        pred_group = np.array(pred_probs[start_idx:end_idx]) + epsilon

        M = 0.5 * (true_group + pred_group)

        kl_p_m = np.sum(true_group * np.log(true_group / M))
        kl_q_m = np.sum(pred_group * np.log(pred_group / M))

        js_div = 0.5 * kl_p_m + 0.5 * kl_q_m
        js_divs.append(js_div)

        start_idx = end_idx

    return np.mean(js_divs)

def answer_extractor(response):
    response = str(response).replace("*","").replace("\n"," ").replace("\r"," ")
    try:
        is_just_a_number = float(response.strip())
        return is_just_a_number
    except:
        pass
    response = " "+response.split("rounded from")[0] + ". "
    pattern = r'(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)'
    numbers = []
    for match in re.finditer(pattern, response):
        num_str = match.group(1)
        start = match.start(1)
        end = match.end(1)

        if end < len(response) and response[end] == '%':
            num = float(num_str) / 100
            numbers.append(num)
        elif start > 0 and response[start - 1] == '%':
            num = float(num_str) / 100
            numbers.append(num)
        elif response[start - 1] == '.' and response[start - 2] == ' ':
            numbers.append(float("0." + num_str))
        elif response[start - 1] == ' ' and (response[end] == ' ' or response[end] == '.' or response[end] == '\n' or response[end] == '*' or end == len(response)):
            numbers.append(float(num_str))
    numbers = [num for num in numbers if num <= 1]
    if numbers[-1]<0:
        print("Error: Negative number",numbers)
    return numbers[-1]

def gumbel_max_normalize_groups(values, groups):
    normalized_values = []
    start_idx = 0
    while start_idx < len(values):
        current_group = groups[start_idx]
        end_idx = start_idx
        while end_idx < len(groups) and groups[end_idx] == current_group:
            end_idx += 1
        group_values = values[start_idx:end_idx]
        max_value = max(group_values)
        exp_values = [math.exp(v - max_value) for v in group_values]
        exp_sum = sum(exp_values)

        if exp_sum != 0:
            normalized_values.extend([ev / exp_sum for ev in exp_values])
        else:
            normalized_values.extend([0] * len(group_values))
        start_idx = end_idx
    return normalized_values


def normalize_groups(values, groups):
    normalized_values = []
    start_idx = 0
    while start_idx < len(values):
        current_group = groups[start_idx]
        end_idx = start_idx
        while end_idx < len(groups) and groups[end_idx] == current_group:
            end_idx += 1
        group_values = values[start_idx:end_idx]
        group_sum = sum(group_values)
        if group_sum != 0:
            normalized_values.extend([v / group_sum for v in group_values])
        else:
            #print("Error: Group sum is 0", [1/len(group_values)] * len(group_values))
            normalized_values.extend([1/len(group_values)] * len(group_values))
        start_idx = end_idx
    return normalized_values


def print_metric_tables(results, metrics,names):
    for parent in sorted(results.keys()):
        table_data = []
        headers = ["dataset", "Model", "Normalization method"] + [f"{metric} " for metric in metrics]
        for data_set_index in range(0,len(names)):
            dataset_name = names[data_set_index]
            for model in results[parent][dataset_name]:
                for norm_method, values in results[parent][dataset_name][model].items():
                    row = [dataset_name, model, norm_method]
                    for metric in metrics:
                        mean_value = np.mean(values[metric])
                        row.append(f"{mean_value:.4f}")
                    table_data.append(row)

        print(f"\nMetrics Table - Parents: {parent}")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\n")


def normalize_groups_powered(values, groups, power=1):
    normalized_values = []
    start_idx = 0
    while start_idx < len(values):
        current_group = groups[start_idx]
        end_idx = start_idx
        while end_idx < len(groups) and groups[end_idx] == current_group:
            end_idx += 1
        group_values = values[start_idx:end_idx]

        transformed_values = [v ** power for v in group_values]
        transformed_sum = sum(transformed_values)

        if transformed_sum != 0:
            normalized_values.extend([v / transformed_sum for v in transformed_values])
        else:
            normalized_values.extend([0] * len(group_values))

        start_idx = end_idx
    return normalized_values

def calculate_metrics(predicted_answers,true_answers,same_CPT,results,names,data_set_index,model_name,results_df,node_names,cathegorization_factor_name):

    for i in sorted(predicted_answers.keys()):
        true_entropy = calculate_entropy(true_answers[i])
        raw_r2 = r2_score(true_answers[i], predicted_answers[i])

        raw_kl = calculate_kl_divergence(true_answers[i], predicted_answers[i], same_CPT[i], node_names[i],normalized=False)
        raw_JS = Jensen_Shannon_divergence(true_answers[i], predicted_answers[i], same_CPT[i], node_names[i],normalized=False)
        results[i][names[data_set_index]][model_name]["Raw"]["R2 Score"].append(raw_r2)
        results[i][names[data_set_index]][model_name]["Raw"]["KL Divergence"].append(raw_kl)
        results_df.append({
            'dataset': names[data_set_index],
            'Model': model_name,
            'Normalization method': 'Raw',
            'R2 Score': raw_r2,
            'CPT KL Divergence': raw_kl,
            'CPT JS Divergence': raw_JS,
            cathegorization_factor_name: i,
            'entropy': true_entropy,
            'predicted_entropy': calculate_entropy(predicted_answers[i])
        })

        normalized_preds = normalize_groups(predicted_answers[i], same_CPT[i])
        norm_r2 = r2_score(true_answers[i], normalized_preds)
        norm_kl = calculate_kl_divergence(true_answers[i], normalized_preds, same_CPT[i], node_names[i])
        norm_JS = Jensen_Shannon_divergence(true_answers[i], normalized_preds, same_CPT[i], node_names[i])
        results[i][names[data_set_index]][model_name]["Normalized"]["R2 Score"].append(norm_r2)
        results[i][names[data_set_index]][model_name]["Normalized"]["KL Divergence"].append(norm_kl)
        results_df.append({
            'dataset': names[data_set_index],
            'Model': model_name,
            'Normalization method': 'Normalized',
            'R2 Score': norm_r2,
            'CPT KL Divergence': norm_kl,
            'CPT JS Divergence': norm_JS,
            cathegorization_factor_name: i,
            'entropy': true_entropy,
            'predicted_entropy': calculate_entropy(normalized_preds)
        })


def entire_BN_KL_divergence(model_GT,model_Predicted):
    infer_GT, infer_Predicted = VariableElimination(model_GT), VariableElimination(model_Predicted)
    total_kl = 0.0

    for node in model_Predicted.nodes():
        cpd_Predicted, cpd_GT = model_Predicted.get_cpds(node), model_GT.get_cpds(node)
        parents = cpd_Predicted.variables[1:]
        if not parents:
            kl = np.sum(cpd_GT.values * np.log((cpd_GT.values + epsilon) / (cpd_Predicted.values + epsilon)))
            total_kl += kl
            continue

        prob_parents = infer_GT.query(variables=parents, evidence={})
        kl_array = np.sum(cpd_GT.values * np.log((cpd_GT.values + epsilon) / (cpd_Predicted.values + epsilon)), axis=0)
        total_kl += np.sum(prob_parents.values * kl_array)
    return total_kl



def create_mixture_model(model1, model2, lam=0.5):
    averaged_model = BayesianNetwork(model1.edges())
    averaged_model.add_nodes_from(model1.nodes())

    infer1, infer2 = VariableElimination(model1), VariableElimination(model2)

    for node in model1.nodes():
        cpd1, cpd2 = model1.get_cpds(node), model2.get_cpds(node)

        if not model1.get_parents(node):
            new_values = []
            for i in range(cpd1.variable_card):
                val = lam * cpd1.values[i] + (1 - lam) * cpd2.values[i]
                new_values.append([val])
            averaged_cpd = TabularCPD(
                variable=cpd1.variable,
                variable_card=cpd1.variable_card,
                values=new_values
            )
        else:
            parents = model1.get_parents(node)
            evidence_card = [len(states) for states in [cpd1.state_names[parent] for parent in model1.get_parents(node)]]  # list of cardinals for each parent (in the order of cpd1.evidence)
            num_cols = int(np.prod(evidence_card))
            # We'll fill a new table with shape (variable_card, num_cols)
            new_values = np.zeros((cpd1.variable_card, num_cols))

            # Compute the joint marginals on the parents for each model.
            # These are DiscreteFactor objects that allow lookup of probability for a given assignment.
            marg1 = infer1.query(variables=parents, joint=True)
            marg2 = infer2.query(variables=parents, joint=True)

            # Iterate over all possible configurations of the parents.
            # We use the ordering implied by evidence_card.
            col_index = 0
            # For each parent, get the list of state names from the CPD.
            parents_state_names = [cpd1.state_names[p] for p in parents]
            for idx_tuple in itertools.product(*[range(card) for card in evidence_card]):
                # Build an evidence dictionary mapping parent names to state names.
                evidence_assignment = {}
                for j, p in enumerate(parents):
                    evidence_assignment[p] = parents_state_names[j][idx_tuple[j]]

                # Get P(pa) from both models.
                p1_pa = marg1.get_value(**evidence_assignment)
                p2_pa = marg2.get_value(**evidence_assignment)
                p_mix_pa = lam * p1_pa + (1 - lam) * p2_pa

                # For each state of the node, combine the conditional probabilities.
                # cpd1.values and cpd2.values are assumed to be arranged so that the column index
                # matches the ordering given by the Cartesian product of evidence states.
                for i in range(cpd1.variable_card):
                    p1_cond = cpd1.values.reshape(cpd1.variable_card,-1)[i][col_index]
                    p2_cond = cpd2.values.reshape(cpd1.variable_card,-1)[i][col_index]
                    # Joint probability for x and the current evidence configuration.
                    joint = lam * (p1_cond * p1_pa) + (1 - lam) * (p2_cond * p2_pa)
                    # Divide by the mixed marginal for the evidence (if nonzero).
                    new_values[i, col_index] = joint / p_mix_pa if p_mix_pa > 0 else 0
                col_index += 1

            averaged_cpd = TabularCPD(
                variable=cpd1.variable,
                variable_card=cpd1.variable_card,
                values=new_values.tolist(),
                evidence=parents,
                evidence_card=evidence_card
            )
        try:
            averaged_model.add_cpds(averaged_cpd)
        except Exception as e:
            print("Error: CPD could not be added for node", node, e)

    return averaged_model


def calculate_BN_KL_divergence(model_GT,file_name,JSD=False):
    initial_cpts = load_initial_cpts(file_name)
    initial_cpts_used = {}
    for node in model_GT.nodes():
        parents = list(model_GT.get_parents(node))
        if not parents:
            if node in initial_cpts:
                initial_cpts_used[node] = dict(initial_cpts[node])
        else:
            if node in initial_cpts:
                converted_initial_cpt = {}
                for parent_tuple, child_dict in initial_cpts[node].items():
                    parent_values_only = tuple(value for _, value in parent_tuple)
                    converted_initial_cpt[parent_values_only] = dict(child_dict)
                initial_cpts_used[node] = converted_initial_cpt
    model_Predicted = save_model_as_bif(model_GT, initial_cpts_used,"","",just_return_it=True)

    if JSD:
        M_model = create_mixture_model(model_GT, model_Predicted)
        JSD_value = 0.5 * (entire_BN_KL_divergence(model_GT,M_model) + entire_BN_KL_divergence(model_Predicted,M_model))
        if JSD_value>2 and len(model_GT.nodes)<10:
            print("Error: JSD value is too high", JSD_value)
        return JSD_value
    return entire_BN_KL_divergence(model_GT,model_Predicted)