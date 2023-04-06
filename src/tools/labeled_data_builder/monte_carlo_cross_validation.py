"""Method to generate simulated samples from a dataset, using Monte-Carlo Cross-Validation."""

from random import sample
from typing import Tuple

import pandas as pd


def generate_train_test_sample(data: pd.DataFrame, train_percentage: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a new simulated sample from the given data, using Monte-Carlo Cross-Validation. The generated sample
    differentiates a new random partition between train and test data, the entire original dataset is used in the
    generated sample (the sample is of the same size as the original DataFrame).

    Args:
        data (pd.DataFrame): The original data to sample from.
        train_percentage (float): Proportion of labeled data going into the train dataset, between 0 and 1 inclusive.

    Returns:
        train_sample (pd.DataFrame): The generated train subset sample.
        test_sample (pd.DataFrame): The generated test subset sample.

    """

    if not 0 <= train_percentage <= 1:
        raise ValueError("Parameter 'train_percentage' must be a number between 0 and 1 inclusive.")

    test_size = int((1 - train_percentage) * len(data))
    test_index = sample(list(range(len(data))), test_size)
    train_data = data.iloc[[i for i in range(len(data)) if i not in test_index]]
    test_data = data.iloc[test_index]

    return train_data, test_data
