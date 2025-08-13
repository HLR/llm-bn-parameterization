import re, ast, os, random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

def read_names_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        return df['name'].tolist(),df['explanation_dict'].tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []


def create_text(model, grouped=False):
    """
    Takes in a pgmpy model.
    If grouped=True, returns a dictionary grouping states by their parent conditions and a list of possible states for each node.

    Args:
        model: A pgmpy BayesianNetwork model
        grouped: Boolean indicating whether to return grouped format (default: False)

    Returns:
        If grouped=False: String with textual explanation
        If grouped=True: Tuple of (dict of grouped states, dict of possible states)
    """
    context = ""
    grouped_states = {}
    node_possible_states = {}

    cpds = model.get_cpds()
    for cpd in cpds:
        variable = cpd.variable
        parents = model.get_parents(variable)
        state_names = cpd.state_names
        variable_states = state_names[variable]
        parent_states = {parent: state_names[parent] for parent in parents}
        node_possible_states[variable] = variable_states
        if not parents:
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
            import itertools
            parent_state_names = [state_names[parent] for parent in parents]
            parent_state_combinations = list(itertools.product(*parent_state_names))
            variable_state_names = state_names[variable]

            var_indices = range(len(variable_state_names))
            for parent_state_combo in parent_state_combinations:
                parent_state_dict = dict(zip(parents, parent_state_combo))
                parent_indices = [state_names[parent].index(state) for parent, state in parent_state_dict.items()]
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
    return context.split("\n")[0:-1], node_possible_states

def extract_dict_from_text(text):
    text=text.replace("\\n", "\n").replace('\\"', '"')
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match:
        dict_str = match.group(1).strip()
        if "= {" in dict_str:
            dict_str = "{" + dict_str.split("= {")[1]
        info = ast.literal_eval(dict_str)
    else:
        print("couldn't parse the dictionary about the nodes.")
        return {}
    return info

def make_info_into_str(info):
    if isinstance(info, dict):
        return " ".join([f"{k}: {v}" for k, v in info.items()])
    if isinstance(info, tuple):
        return "description: " + info[0] + " values: " + str(info[1])
    return info

def prepare_IO(i,info,ignore_context=False,is_random=False,is_tokenprob=False,is_inoneprompt=False):
    question = i.split('with probability of')[0] + "?"
    extra_information = "These nodes are related to the question inside a Bayesian Network: \n"
    matches = set(re.findall(r'\b(\w+)\s+is\b', question))
    if not is_random:
        for m in matches: extra_information += m + ":\n" + make_info_into_str(info[m]) + "\n"
    if ignore_context or is_random: extra_information = ""
    if is_random:
        question = question.replace(question.rsplit(" is ", 1)[0].split()[-1],random.choice(list(info.keys())))
    instruction = extra_information + "\nGiven this information answer the following question by providing a probability from 0 to 1 based on your best guess ( you need to make a lot of estimations since the given information is limited). Your answer should include your reasoning and at the end a sentence that says 'The probability of the question is: ' followed by the probability."
    if " then " in question: node_name=question.rsplit(" is ",1)[0].rsplit(" then ",1)[1].strip()
    else: node_name=question.rsplit(" is ",1)[0].strip()

    if is_tokenprob:
        question = question.rsplit(" is ",1)[0] + " is ?"
        question = question.lower()
        instruction = extra_information + "\n Given this information, answer the following question by providing the most probable state of the node. Your answer should only include the state of the node without capitalization, space, or any additional information."
        instruction = instruction.lower()

    if is_inoneprompt:
        question = question.rsplit(" is ",1)[0] + " ?"
        question = question.replace(" then "," then what is the probability distribution of ")
        instruction = extra_information + "\n Given this information, answer the following question by providing the probability distribution of the node. Your answer should include your reasoning and, at the end, a sentence that says 'The probability distribution of the node is: ' followed by the probabilities given in a tuple with each probability representing a state in the given order."

    return question, instruction, node_name

def save_llm_answer(file_name, result, i):
    df = pd.DataFrame([{"raw_text": result, "answer": i.split(' of ')[1].strip(), "question": i.split(' of ')[0].strip()}])
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, index=False)

def answer_extractor(response,return_tuple_size=0):
    response = str(response).replace("*","").replace("\n"," ").replace("\r"," ")
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
    if return_tuple_size == 0:
        return numbers[-1]
    return tuple(numbers[-return_tuple_size:])

def categorize_entropy(value):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for i in range(len(bins) - 1):
        if bins[i] <= value <= bins[i + 1]:
            return f"{bins[i]}-{bins[i + 1]}"
    return "1.0+"

def get_display_name(model_name: str) -> str:
    """
    Map an internal model identifier to a concise, human‑readable label.
    Handles suffixes such as `withoutcontext`, `random`, `tokenprob`,
    and `inoneprompt`, plus new families `o3` and `o3‑mini`.
    """
    # Simple one‑to‑one overrides
    simple = {"Random": "Random", "Mean": "Uniform"}
    if model_name in simple:
        return simple[model_name]

    # MLE‑style Data‑X → MLE‑X
    if model_name.startswith("Data-"):
        return f"MLE-{model_name.split('-', 1)[1]}"

    name = model_name.lower()

    # Family → pretty base‑name
    families = {
        "gpt-4o-mini": "GPT-4o-mini",
        "gpt-4o":      "GPT-4o",
        "o3-mini":     "O3-mini",
        "o3":          "O3",
        "gemini-pro":  "Gemini Pro",
        "claude-3.5":  "Claude 3.5",
        "deepseek":    "DeepSeek",          # will be overridden for R1 below
    }

    base = next((pretty for key, pretty in families.items() if key in name), None)
    if base is None:        # unknown – fall back to raw string
        return model_name

    # Special‑case DeepSeek Reasoning (R1)
    if base == "DeepSeek" and "r1" in name:
        base = "DeepSeek-R1"
    elif base == "DeepSeek":
        base = "DeepSeek-V3"

    # Common variant suffixes
    if "withoutcontext" in name:
        variant = "No Context"
    elif "random" in name:
        variant = "Random"
    elif "tokenprob" in name:
        variant = "Token Probability"
    elif "inoneprompt" in name:
        variant = "FullDist"
    else:
        variant = "SepState"

    return f"{base} ({variant})"

def create_custom_legend(ax, palette, sorted_models,args):
    # Remove the default legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    # Create custom legend handles with logical grouping
    legend_handles = []
    # Group Data models
    data_models = [m for m in sorted_models if "Data-" in m]
    # Group GPT models
    gpt_models = [m for m in sorted_models if "gpt" in m.lower() and "Data-" not in m]
    # Other models
    other_models = [m for m in sorted_models if "Data-" not in m and "gpt" not in m.lower()]
    if data_models:
        data_color = palette[data_models[0]]
        data_patch = mpatches.Patch(color=data_color, label="Data Models")
        legend_handles.append(data_patch)
    for model in gpt_models:
        model_patch = mpatches.Patch(color=palette[model], label=get_display_name(model))
        legend_handles.append(model_patch)
    for model in other_models:
        model_patch = mpatches.Patch(color=palette[model], label=get_display_name(model))
        legend_handles.append(model_patch)
    plt.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.4),
        ncol=5,
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=args.legend_fontsize
    )

def create_feature_data(args, feature=""):
    result_file_name = 'results/llm_evaluation_results_BN Models.csv'
    if "Number of States" in feature:
        result_file_name = 'results/llm_evaluation_results_Number of States.csv'
    if "Number of Parents" in feature:
        result_file_name = 'results/llm_evaluation_results_Number of Parents.csv'

    llm_data = pd.read_csv(result_file_name)
    networks_data = pd.read_csv('../PreprocessingBayesianNetworks/valid_networks.csv')
    normalized_llm_data = llm_data[llm_data['Normalization method'] == 'Normalized']
    merged_data = pd.merge(normalized_llm_data, networks_data, left_on='dataset', right_on='Name')
    merged_data['entropy_category'] = merged_data['entropy'].apply(categorize_entropy)
    merged_data['predicted_entropy_category'] = merged_data['predicted_entropy'].apply(categorize_entropy)
    feature_data = merged_data[merged_data['Model'].isin(args.models)].copy()
    return feature_data

def tuple_extractor(input_text):
    pattern = r"\(\s*([0-9]*\.?[0-9]+\s*%?(?:\s*,\s*[0-9]*\.?[0-9]+\s*%?)*)\s*\)"
    matches = re.findall(pattern, input_text)
    results = []
    for match in matches:
        items = match.split(',')
        tuple_values = []
        for item in items:
            item = item.strip()
            if item.endswith('%'):
                numeric_str = item[:-1].strip()
                value = float(numeric_str) / 100.0
            else:
                value = float(item)
            tuple_values.append(value)
        results.append(tuple(tuple_values))
    return results[-1]

def create_model_palette(models, colormap_name='Set2'):
    model_types = {}
    print(models)
    for model in models:
        if "data-" in get_display_name(model).lower():
            model_types["data Models"] = model_types.get("data Models", []) + [model]
        elif "sepstate" in get_display_name(model).lower():
            model_types["sepstate Models"] = model_types.get("sepstate Models", []) + [model]
        elif "fulldist" in get_display_name(model).lower():
            model_types["fulldist Models"] = model_types.get("fulldist Models", []) + [model]
        elif "random" in get_display_name(model).lower():
            model_types["random Models"] = model_types.get("random Models", []) + [model]
        elif "no context" in get_display_name(model).lower():
            model_types["no context Models"] = model_types.get("no context Models", []) + [model]
        elif "token probability" in get_display_name(model).lower():
            model_types["token probability Models"] = model_types.get("token probability Models", []) + [model]
        else:
            model_types["Other Models"] = model_types.get("Other Models", []) + [model]
    palette = {}
    colors = sns.color_palette(colormap_name)
    # All Data models get shades of the same solid color - no fading
    for num,model_group in enumerate(["data Models", "sepstate Models", "fulldist Models", "random Models", "no context Models", "token probability Models"]):
        if model_group in model_types:
            data_models = model_types[model_group]
            for model in data_models:
                palette[model] = colors[num]

    color_idx = len(["data Models", "sepstate Models", "fulldist Models", "random Models", "no context Models", "token probability Models"])
    for model_type, models_list in model_types.items():
        if model_type in ["data Models", "sepstate Models", "fulldist Models", "random Models", "no context Models", "token probability Models"]:
            continue  # Already handled
        for model in models_list:
            if model == "Mean":
                palette[model] = "white"
                continue
            palette[model] = colors[color_idx]
            color_idx += 1

    return palette

model_name_translation = {"GPT-4o (SepState)":"gpt-4o", "GPT-4o (FullDist)":"gpt-4o-inoneprompt", "GPT-4o (Token Probability)":"gpt-4o-tokenprob", "GPT-4o (Without Context)":"gpt-4o-withoutcontext", "GPT-4o (Random)":"gpt-4o-random",
    "GPT-4o-mini (SepState)":"gpt-4o-mini", "GPT-4o-mini (FullDist)":"gpt-4o-mini-inoneprompt", "GPT-4o-mini (Token Probability)":"gpt-4o-mini-tokenprob", "GPT-4o-mini (Without Context)":"gpt-4o-mini-withoutcontext", "GPT-4o-mini (Random)":"gpt-4o-mini-random",
    "O3 (SepState)":"o3", "O3 (FullDist)":"o3-inoneprompt",
    "O3-mini (SepState)":"o3-mini", "O3-mini (FullDist)":"o3-mini-inoneprompt",
    "DeekSeek-R1 (SepState)":"deepseek-R1", "DeekSeek-R1 (FullDist)":"deepseek-R1-inoneprompt",
    "DeekSeek-V3 (SepState)":"deepseek", "DeekSeek-V3 (FullDist)":"deepseek-inoneprompt", "DeekSeek-V3 (Token Probability)":"deepseek-tokenprob", "DeekSeek-V3 (Without Context)":"deepseek-withoutcontext", "DeekSeek-V3 (Random)":"deepseek-random",
    "Uniform":"Mean","Random":"Random",
    "Gemini Pro (SepState)":"gemini-pro","Claude 3.5 (SepState)":"claude-3.5","Gemini Pro (FullDist)":"gemini-pro-inoneprompt","Claude 3.5 (FullDist)":"claude-3.5-inoneprompt"
}
