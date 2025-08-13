import re, os
import pandas as pd
from collections import defaultdict
pd.set_option('display.precision', 30)

def answer_extractor(response):
    response = str(response)
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
    return numbers[-1]

def load_initial_cpts(cpt_file, GT=False):

    if not os.path.exists(cpt_file):
        print(f"ERROR: File {cpt_file} does not exist.")
        return {}

    initial_cpts = {}
    df = pd.read_csv(cpt_file)

    df['answer_prob'] = df['raw_text'].apply(answer_extractor).clip(lower=0.0001)
    df['answer'] = df['answer_prob'].apply(lambda x: "0.01%." if x==0.0001 else str(x * 100) + "%.")
    conditional_pattern = re.compile \
        (r"If\s+(.*?)\s*,\s*then\s+([A-Za-z0-9_-]+)\s+is\s+(\S+)\s+with\s+probability\s+([\d.]+)%" ,re.IGNORECASE)
    unconditional_pattern = re.compile(r"^([A-Za-z0-9_-]+)\s+is\s+(\S+)\s+with\s+probability\s+([\d.]+)%" ,re.IGNORECASE)
    unconditional_probs = defaultdict(dict)
    conditional_probs = defaultdict(lambda: defaultdict(dict))

    for idx, row in df.iterrows():
        question_str = str(row['question'])
        answer_str = str(row['answer'])
        line = f"{question_str} {answer_str}".strip()
        if not line: continue

        cond_match = conditional_pattern.match(line)
        if cond_match:
            conditions_str, node, state, prob_str = cond_match.groups()
            prob = float(prob_str) / 100.0
            cond_parts = [c.strip() for c in conditions_str.split(" and ")]
            parent_values = []
            for part in cond_parts:
                pv_match = re.match(r"^([A-Za-z0-9_-]+)\s+is\s+(\S+)$", part.strip(), re.IGNORECASE)
                if pv_match:
                    p_node, p_state = pv_match.groups()
                    parent_values.append((p_node.strip(), p_state.strip()))
                else:
                    print(f"ERROR: Could not parse parent value from '{part}' in line '{line}'")
            parent_values_tuple = tuple(parent_values)
            conditional_probs[node][parent_values_tuple][state] = prob if not GT else float(answer_str.strip(" .%")) / 100
            continue

        uncond_match = unconditional_pattern.match(line)
        if uncond_match:
            node, state, prob_str = uncond_match.groups()
            prob = float(prob_str) / 100.0
            unconditional_probs[node][state] = prob if not GT else float(answer_str.strip(" .%")) / 100
            continue

    for node, states_probs in unconditional_probs.items():
        total = sum(states_probs.values())
        if total > 0:
            for s in states_probs:
                states_probs[s] = states_probs[s] / total
        else:
            for s in states_probs:
                states_probs[s] = 1.0 / len(states_probs)
        initial_cpts[node] = states_probs

    for node, parent_map in conditional_probs.items():
        for pv_tuple, state_probs in parent_map.items():
            total = sum(state_probs.values())
            if total > 0:
                for s in state_probs:
                    state_probs[s] = state_probs[s] / total
            else:
                for s in state_probs:
                    state_probs[s] = 1.0 / len(state_probs)
        initial_cpts[node] = dict(parent_map)

    return initial_cpts