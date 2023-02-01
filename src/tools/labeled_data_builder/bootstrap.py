"""Method(s) to generate simulated samples from a dataset, using bootstrapping."""

from random import randint

import pandas as pd


def generate_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Generate a new simulated sample from the given data, using bootstrapping. The generated sample has the same size
    as the original data and each row is sampled with replacement.

    Args:
        data (pd.DataFrame): The original data to sample from.

    Returns:
        sample (pd.DataFrame): The generated sample.

    """
    data_length = len(data)
    sample = pd.DataFrame(columns=data.columns)
    while len(sample) < data_length:
        i = randint(0, data_length - 1)
        sample = pd.concat([sample, data.iloc[i : i + 1]])
    return sample
