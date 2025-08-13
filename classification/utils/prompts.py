import json, textwrap
import os, json, itertools, sys, numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field

def build_labeling_prompt(
    *,
    name: str,
    description: str,
    target_col: str,
    target_values: List[Any],
    row: pd.Series,
    with_cot: bool
) -> str:
    """
    Return an instruction-rich prompt for classifying a single row.
    If `with_cot` is True, the model must show its reasoning and then
    output `Answer: <value>` on a new line.
    Otherwise, it must output ONLY the predicted value.
    """
    # Exclude the target column from features
    features = {k: v for k, v in row.items() if k != target_col}

    cot_rules = (
        "RULES:\n"
        "  1. Think step-by-step in plain text (no markdown) to decide which "
        f"value best fits the row.\n"
        "  2. On a NEW line after your reasoning, output exactly:\n"
        '     Answer: <value>\n'
        "  3. <value> must be one of " + json.dumps(target_values) + ".\n"
        "  4. Do NOT output anything else after the answer line."
    )

    direct_rules = (
        "RULES:\n"
        "  1. Output ONLY the predicted value—no explanation, no quotes, "
        "no extra text.\n"
        "  2. The value must be one of " + json.dumps(target_values) + "."
    )

    prompt = f"""
    ROLE: You are a data-labeling assistant.

    DATA:
      • Dataset name: {name}
      • Description : {description or 'N/A'}
      • Row (JSON)  :
        {json.dumps(features, default=str)}

    TASK: Predict the correct value for the target column “{target_col}”.

    {(cot_rules if with_cot else direct_rules)}

    SELF-CHECK: Before finalising, verify you obeyed every rule.
    """
    return textwrap.dedent(prompt).strip()

def build_HC_fulldist_prompt(chat_model):

    class RowConditionalProbability(BaseModel):
        probabilities: Dict[str, float] = Field(
            description="Map each possible value of the child variable to its probability. Must sum to 1.")

    row_parser = PydanticOutputParser(pydantic_object=RowConditionalProbability)
    row_fix_parser = OutputFixingParser.from_llm(parser=row_parser, llm=chat_model)

    row_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a meticulous Bayesian statistician.  Given the dataset description and a *specific* parent assignment, "
            "return ONLY the conditional distribution of the child variable in **valid JSON** following:\n{format_instructions}"
        ),
        (
            "human",
            "Dataset: {dataset_name}\n\n"
            "Description: {dataset_desc}\n\n"
            "Child variable: {child}\nPossible values: {child_values}\n\n"
            "Parent assignment (condition): {parent_assignment}\n\n"
            "Provide P({child} | {parent_assignment})."
        ),
    ]).partial(format_instructions=row_parser.get_format_instructions())
    return row_prompt, row_fix_parser

def build_NB_FullDist_prior_prompt(chat_model):

    class PriorProbability(BaseModel):
        """Model for prior probability distribution of target values."""
        probabilities: Dict[str, float] = Field(
            description="Dictionary mapping target values to their probabilities. Probabilities must sum to 1."
        )

    prior_parser = PydanticOutputParser(pydantic_object=PriorProbability)
    prior_fixing_parser = OutputFixingParser.from_llm(parser=prior_parser, llm=chat_model)

    # Query LLM for prior probabilities
    prior_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a statistical expert. Given a dataset description, estimate the prior probabilities "
         "for each possible value of the target variable. Return ONLY a JSON object with the following format:\n\n"
         "{format_instructions}"
         ),
        ("human",
         "Dataset: {dataset_name}\n\n"
         "Description: {dataset_desc}\n\n"
         "Target variable: {target_var}\n"
         "Possible values: {target_values}\n\n"
         "Estimate the prior probability distribution for the target variable."
         )
    ]).partial(format_instructions=prior_parser.get_format_instructions())

    return prior_prompt, prior_fixing_parser

def build_NB_FullDist_dependant_prompt(chat_model):

    class SingleConditionalProbability(BaseModel):
        """Distribution of a feature given ONE target value."""
        probabilities: Dict[str, float] = Field(
            description="Map each feature value to P(feature_value | target = THIS value). Must sum to 1."
        )

    single_parser = PydanticOutputParser(pydantic_object=SingleConditionalProbability)
    single_fix_parser = OutputFixingParser.from_llm(parser=single_parser, llm=chat_model)

    column_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a statistical expert.  Given a dataset description, estimate the conditional "
         "probabilities for a feature **given a single target value**. "
         "Return ONLY a JSON object with the following format:\n{format_instructions}"
         ),
        ("human",
         "Dataset: {dataset_name}\n\n"
         "Description: {dataset_desc}\n\n"
         "Target variable: {target_var}\n"
         "Current target value: {target_value}\n\n"
         "Feature: {feature}\n"
         "Possible feature values: {feature_values}\n\n"
         "Estimate the probability distribution P({feature} | {target_var} = {target_value})."
         )
    ]).partial(format_instructions=single_parser.get_format_instructions())

    return column_prompt, single_fix_parser

def build_NB_SepState_prior_prompt(chat_model):
    """
    Build a prompt for querying individual prior probabilities instead of full distributions.
    """
    class SinglePriorProbability(BaseModel):
        """Model for a single prior probability value."""
        probability: float = Field(
            description="The prior probability value for the specific target value."
        )

    prior_parser = PydanticOutputParser(pydantic_object=SinglePriorProbability)
    prior_fixing_parser = OutputFixingParser.from_llm(parser=prior_parser, llm=chat_model)

    # Query LLM for a single prior probability
    prior_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a statistical expert. Given a dataset description, estimate the prior probability "
         "for a specific value of the target variable. Return ONLY a JSON object with the following format:\n\n"
         "{format_instructions}"
         ),
        ("human",
         "Dataset: {dataset_name}\n\n"
         "Description: {dataset_desc}\n\n"
         "Target variable: {target_var}\n"
         "Specific target value: {target_value}\n\n"
         "Estimate the prior probability P({target_var} = {target_value})."
         )
    ]).partial(format_instructions=prior_parser.get_format_instructions())

    return prior_prompt, prior_fixing_parser

def build_NB_SepState_dependant_prompt(chat_model):
    """
    Build a prompt for querying individual conditional probabilities instead of full distributions.
    """
    class SingleValueConditionalProbability(BaseModel):
        """Probability of a specific feature value given a specific target value."""
        probability: float = Field(
            description="The probability value P(feature_value | target_value)."
        )

    single_parser = PydanticOutputParser(pydantic_object=SingleValueConditionalProbability)
    single_fix_parser = OutputFixingParser.from_llm(parser=single_parser, llm=chat_model)

    column_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a statistical expert. Given a dataset description, estimate the conditional "
         "probability for a specific feature value given a specific target value. "
         "Return ONLY a JSON object with the following format:\n{format_instructions}"
         ),
        ("human",
         "Dataset: {dataset_name}\n\n"
         "Description: {dataset_desc}\n\n"
         "Target variable: {target_var}\n"
         "Current target value: {target_value}\n\n"
         "Feature: {feature}\n"
         "Specific feature value: {feature_value}\n\n"
         "Estimate the probability P({feature} = {feature_value} | {target_var} = {target_value})."
         )
    ]).partial(format_instructions=single_parser.get_format_instructions())

    return column_prompt, single_fix_parser