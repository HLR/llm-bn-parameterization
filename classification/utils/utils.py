from pgmpy.factors.discrete import TabularCPD
from typing import Dict, Any, List, Tuple
from sklearn.metrics import f1_score
import os, sys, pandas as pd, numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PreprocessingBayesianNetworks.save_model_pickles import safe_load

NAIVE_BAYES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NaiveBayes_Data")
os.makedirs(NAIVE_BAYES_DIR, exist_ok=True)
epsilon = 1e-4

def create_naive_bayes_network(
    target_col: str,
    features: List[str],
    target_values: List[str],
    feature_values: Dict[str, List[str]],
    cpts: Dict[str, np.ndarray],
    priors: Dict[str, float]
) -> BayesianNetwork:
    """
    Create a Naive Bayes network with the target variable as the parent node
    and all features as child nodes.

    Args:
        target_col: The name of the target variable
        features: List of feature names
        target_values: List of possible values for the target variable
        feature_values: Dictionary mapping feature names to their possible values
        cpts: Dictionary mapping feature names to their conditional probability tables
        priors: Dictionary mapping target values to their prior probabilities

    Returns:
        A BayesianNetwork object representing the Naive Bayes model
    """

    edges = [(target_col, feature) for feature in features]
    model = BayesianNetwork(edges)

    target_cpd = TabularCPD(
        variable=target_col,
        variable_card=len(target_values),
        values=[[priors[str(val)]] for val in target_values],
        state_names={target_col: target_values}
    )
    model.add_cpds(target_cpd)

    for feature in features:
        feature_vals = feature_values[feature]
        feature_cpd = TabularCPD(
            variable=feature,
            variable_card=len(feature_vals),
            values=cpts[feature],
            evidence=[target_col],
            evidence_card=[len(target_values)],
            state_names={
                feature: feature_vals,
                target_col: target_values
            }
        )
        model.add_cpds(feature_cpd)

    model.check_model()
    return model

def infer_target(model: BayesianNetwork, evidence: Dict[str, Any], target_col: str) -> str:
    """
    Use variable elimination to infer the most likely value of the target variable.

    Args:
        model: The Bayesian Network model
        evidence: Dictionary mapping feature names to their observed values
        target_col: The name of the target variable

    Returns:
        The most likely value of the target variable
    """
    inference = VariableElimination(model)
    result = inference.query(variables=[target_col], evidence=evidence)

    # Get the most likely value
    values = result.values
    target_values = model.get_cpds(target_col).state_names[target_col]
    max_index = np.argmax(values)

    return target_values[max_index]


def _f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    legal = y_true.unique()
    if len(legal) == 2: return f1_score(y_true, y_pred, average="binary", pos_label=legal[0])
    return f1_score(y_true, y_pred, average="macro", labels=legal)

def load_naive_bayes_models(folder_name):
    """
    Load Bayesian networks from the Naive Bayes Folder.

    Returns:
        dict: Dictionary of Bayesian network models.
    """
    models = {}
    naive_bayes_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name)

    if not os.path.exists(naive_bayes_folder):
        print(f"Error: folder not found at {naive_bayes_folder}")
        return models

    for filename in os.listdir(naive_bayes_folder):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]
            model_path = os.path.join(naive_bayes_folder, filename)
            try:
                models[model_name] = safe_load(model_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")

    return models