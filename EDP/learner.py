import pandas as pd
from utils import read_names_from_csv
from collections import defaultdict
import numpy as np
import networkx as nx
from utils import save_all_model_versions,n_params_from_graph
from learner_utils import load_initial_cpts
from tqdm import tqdm
from PreprocessingBayesianNetworks.save_model_pickles import safe_load
from pgmpy.sampling import BayesianModelSampling
import itertools, math

csv_path = '../PreprocessingBayesianNetworks/bn_node_explanations.csv'
names, info_raw = read_names_from_csv(csv_path)
models = {}
for model_name in tqdm(names):
    models[model_name] = safe_load(f"../PreprocessingBayesianNetworks/BNs/Pickles/{model_name}.pkl")


class BayesianNetworkLearner:
    def __init__(self, sample_num, prior_weight_hyper_parameter, llm_model_name, sample_method, output_dir, save_BNs, learning_method, args):
        if sample_method == 'per_model_by_parameter': self.prior_weight = prior_weight_hyper_parameter / (sample_num - 2)
        else: self.prior_weight = prior_weight_hyper_parameter/(sample_num) * 3 # for lower data multiply by 3
        self.sample_num = sample_num
        self.initial_cpts = {}
        self.llm_model_name = llm_model_name
        self.sample_method = sample_method
        self.output_dir = output_dir
        self.save_BNs = save_BNs
        self.laplace_smoothing = 1e-8
        self.learning_method = learning_method
        self.evaluation_method = 'normal'
        self.certainty_threshold=0.90
        self.args = args

    def load_models(self):
        for bif_file in names:
            model = models[bif_file]
            model.name = bif_file
            self.initial_cpts = load_initial_cpts(
                f"../EstimationofPriorProbabilitiesbyLLMs/dataset_outputs/datasetname_{model.name}_model_name_{self.llm_model_name}.csv")
            yield model

    def learn_cpt_probabilities(self, node_samples, model):

        initial_cpts_used, learned_from_data, combined_cpts = {}, {}, {}

        for node in model.nodes():
            parents = list(model.get_parents(node))
            node_states = list(model.get_cpds(node).state_names[node])
            pairs = node_samples[node]
            if not parents:
                freq = defaultdict(int)
                for parent_assign, val in pairs:
                    freq[val] += 1
                total = sum(freq.values())
                data_cpt = {}
                node_is_certain=False
                for state in node_states:
                    data_cpt[state] = (freq[state] + self.laplace_smoothing) / (total + len(node_states) * self.laplace_smoothing)
                    if freq[state] / total >= self.certainty_threshold:
                        node_is_certain=True
                learned_from_data[node] = data_cpt
                initial_cpts_used[node] = dict(self.initial_cpts[node])
                combined_probs = {}
                for state in node_states:
                    if node_is_certain and self.evaluation_method == 'replace_certain':
                        combined_probs[state] = data_cpt[state]
                    else:
                        if self.learning_method == 'Dirichlet':
                            combined_probs[state] = (1 - self.prior_weight) * data_cpt[state] + self.prior_weight * self.initial_cpts[node][state]
                        else:  # Counting
                            combined_probs[state] = (freq[state] + self.laplace_smoothing + self.prior_weight * self.initial_cpts[node][state] * total) / (total + len(node_states) * self.laplace_smoothing + self.prior_weight * total)
                        # normalization just in case
                for state in node_states:
                    combined_probs[state] /= sum(combined_probs.values())
                combined_cpts[node] = combined_probs

            else:
                cpd_counts = defaultdict(lambda: defaultdict(int))
                parent_value_counts = defaultdict(int)

                for p_assign, val in pairs:
                    parent_values = tuple(p_assign[p] for p in parents)
                    cpd_counts[parent_values][val] += 1
                    parent_value_counts[parent_values] += 1
                node_is_certain=False
                for pvals_tuple, child_counts in cpd_counts.items():
                    total = parent_value_counts[pvals_tuple]
                    for state in node_states:
                        count = child_counts[state]
                        cpd_counts[pvals_tuple][state] = (count + self.laplace_smoothing) / (total + len(node_states) * self.laplace_smoothing)
                        if count / total >= self.certainty_threshold:
                            node_is_certain = True

                parent_state_lists = [model.get_cpds(p).state_names[p] for p in parents]
                for combo in itertools.product(*parent_state_lists):
                    if combo not in cpd_counts:
                        cpd_counts[combo] = {s: 1 / len(node_states) for s in node_states}
                    else:
                        for s in node_states:
                            if s not in cpd_counts[combo]:
                                cpd_counts[combo][s] = 1 / len(node_states)

                learned_from_data[node] = {pvals_tuple: dict(child_probs) for pvals_tuple, child_probs in cpd_counts.items()}
                converted_initial_cpt = {}
                for parent_tuple, child_dict in self.initial_cpts[node].items():
                    converted_initial_cpt[tuple(value for _, value in parent_tuple)] = dict(child_dict)
                initial_cpts_used[node] = converted_initial_cpt

                combined_cpd = {}
                for pvals_tuple, child_counts in cpd_counts.items():
                    total = parent_value_counts[pvals_tuple]
                    combined_probs = {}
                    if node_is_certain and self.evaluation_method == 'replace_certain':
                        for state in node_states:
                            combined_probs[state] = (child_counts[state] + self.laplace_smoothing) / (total + len(node_states) * self.laplace_smoothing)
                    else:
                        if self.learning_method == 'Dirichlet':
                            for state in node_states:
                                combined_probs[state] = (1 - self.prior_weight) * cpd_counts[pvals_tuple][state] + self.prior_weight * converted_initial_cpt[pvals_tuple][state]
                        else:  # Counting
                            for state in node_states:
                                combined_probs[state] = (child_counts[state]*total + self.laplace_smoothing + self.prior_weight * converted_initial_cpt[pvals_tuple][state] * total) / (total + len(node_states) * self.laplace_smoothing + self.prior_weight * total)
                    # normalization just in case
                    for state in node_states:
                        combined_probs[state] /= sum(combined_probs.values())
                    combined_cpd[pvals_tuple] = combined_probs
                combined_cpts[node] = combined_cpd

        if self.save_BNs:
            save_all_model_versions(model, initial_cpts_used, learned_from_data, combined_cpts, self.sample_num,
                                    self.prior_weight, self.llm_model_name, self.sample_method, self.output_dir,create_data_baselines=self.args.create_data_baselines)
        return initial_cpts_used, learned_from_data, combined_cpts

    def generate_samples(self, model, n_samples, sample_method):
        """
        :param sample_method: can be'per_node_and_parents','per_node_and_parents_by_size', 'per_model_by_parameter'
        """
        topological_nodes = list(nx.topological_sort(model))
        node_samples, cpd_dict = {}, {}
        for cpd in model.get_cpds(): cpd_dict[cpd.variable] = cpd
        if sample_method in ['per_node_and_parents', 'per_node_and_parents_by_size']:
            for node in topological_nodes:
                parents = list(model.get_parents(node))
                if not parents:
                    possible_values = cpd_dict[node].state_names[node]
                    probs = cpd_dict[node].values
                    if not sum(probs) > .99: print("Error: the probabilities do not sum to 1")
                    samples_for_node = []
                    multiplication_factor = 1 if sample_method == 'per_node_and_parents' else len(possible_values)
                    for _ in range(n_samples * multiplication_factor):
                        val = np.random.choice(possible_values, p=probs / sum(probs))
                        samples_for_node.append(({}, val))
                    node_samples[node] = samples_for_node
                else:
                    parent_states = []
                    parent_vals_by_node = []
                    for p in parents:
                        parent_vals_by_node.append(cpd_dict[p].state_names[p])

                    for combo in itertools.product(*parent_vals_by_node):
                        parent_assignment = dict(zip(parents, combo))
                        parent_states.append(parent_assignment)

                    samples_for_node = []
                    node_cpd = cpd_dict[node]
                    possible_values = node_cpd.state_names[node]

                    for p_assign in parent_states:
                        reduced_cpd = node_cpd.copy()
                        reduced_cpd = reduced_cpd.to_factor().reduce(
                            [(par, p_assign[par]) for par in parents], inplace=False
                        )
                        distribution = reduced_cpd.values
                        distribution = distribution / np.sum(distribution)
                        multiplication_factor = 1 if sample_method == 'per_node_and_parents' else len(possible_values)
                        for _ in range(n_samples * multiplication_factor):
                            val = np.random.choice(possible_values, p=distribution / sum(distribution))
                            samples_for_node.append((p_assign, val))

                    node_samples[node] = samples_for_node
        elif sample_method == 'per_model_by_parameter':
            sampler = BayesianModelSampling(model)  # forward sampling
            df_samples: pd.DataFrame = sampler.forward_sample(
                size=n_samples,#*n_params_from_graph(model),
                show_progress=False
            )

            # build the same structure you use elsewhere:
            #   node_samples[node] = [ ( {parents:values}, node_value ), ... ]
            for node in topological_nodes:
                node_samples[node] = []

            for _, row in df_samples.iterrows():  # iterate over joint samples
                for node in topological_nodes:
                    parents = list(model.get_parents(node))
                    parent_assignment = {p: row[p] for p in parents}
                    node_samples[node].append((parent_assignment, row[node]))

        else:
            raise NotImplementedError(f"Sampling method {sample_method} not implemented")
        return node_samples

    def _combine_probabilities(self, data_probs, initial_probs, states):
        # (freq[state] + self.laplace_smoothing) / (total + len(node_states) * self.laplace_smoothing)
        combined_probs = {}
        for state in states:
            combined_probs[state] = (1 - self.prior_weight) * data_probs[state] + self.prior_weight * initial_probs[
                state]
        for state in states:
            combined_probs[state] /= sum(combined_probs.values())
        return combined_probs