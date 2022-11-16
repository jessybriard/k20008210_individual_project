"""Class to build labeled data from a DataFrame for time series forecasting."""

import math

import pandas as pd


def create_labeled_data_individual_approach(ticker: str, data: pd.DataFrame, features_length: int) -> pd.DataFrame:
    """Create labeled data for time series forecasting, using given historical data for a ticker, using features_length
    previous values in the time series as features and the next value as label.

    Args:
        ticker (str): The code for the asset we create labeled data for.
        data (pd.DataFrame): The historical time series for the ticker.
        features_length (int): The number of previous binary returns to use as features to predict the next one
            (the label).

    Returns:
        labeled_data (pd.DataFrame): The created labeled data, contains a column for features and a column for label.

    """

    if ticker not in data.columns:
        raise ValueError("Parameter 'ticker' must represent a valid column in the 'data' provided as parameter.")

    if features_length < 1:
        raise ValueError("Parameter 'features_length' must be a strictly positive integer (>= 1).")

    values = list(data[ticker].values)
    features = []
    label = []
    for i in range(len(values) - features_length):
        if True not in [math.isnan(value) for value in values[i : i + features_length + 1]]:
            features.append(values[i : i + features_length])
            label.append(values[i + features_length])
    return pd.DataFrame(data={"features": features, "label": label})
