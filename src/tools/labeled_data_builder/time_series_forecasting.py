"""Methods to build labeled data from a DataFrame for time series forecasting."""

import math
from typing import List

import pandas as pd


def create_labeled_data_individual_approach(ticker: str, data: pd.DataFrame, features_length: int) -> pd.DataFrame:
    """Create labeled data for time series forecasting, using given historical data for a ticker, using features_length
    previous values in the time series as features and the next value as label.

    Args:
        ticker (str): The code for the asset we create labeled data for.
        data (pd.DataFrame): The historical time series for the ticker.
        features_length (int): The number of previous rows to use as features to predict the next one (the label).

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


def create_labeled_data_sector_approach(
    ticker_label: str, tickers_features: List[str], data: pd.DataFrame, features_length: int
) -> pd.DataFrame:
    """Create labeled data for time series forecasting, using given historical data for a ticker, using features_length
    previous values in the time series as features_individual and features_sector and the next value as label. We build
    features for both the individual and sector approach, so that both approaches use the same data points, to allow for
    fairer comparison.

    Args:
        ticker_label (str): The code for the asset we want to predict for (the label).
        tickers_features (List[str]): The codes for the assets we want to use as features_sector for the prediction.
        data (pd.DataFrame): The historical time series for the tickers.
        features_length (int): The number of previous rows to use as features_sector to predict the next one (label).

    Returns:
        labeled_data (pd.DataFrame): The created labeled data, contains a columns for features_individual,
            features_sector and label.

    """

    if ticker_label not in data.columns:
        raise ValueError("Parameter 'ticker_label' must represent a valid column in the 'data' provided as parameter.")

    if not tickers_features or not set(data.columns).issuperset(set(tickers_features)):
        raise ValueError(
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter."
        )

    if features_length < 1:
        raise ValueError("Parameter 'features_length' must be a strictly positive integer (>= 1).")

    if ticker_label not in tickers_features:
        tickers_features.append(ticker_label)

    features_individual = []
    features_sector = []
    label = []
    for i in range(len(data) - features_length):
        features_data_slice = data[tickers_features].iloc[i : i + features_length]
        if not math.isnan(data[ticker_label].iloc[i + features_length]) and True not in [
            math.isnan(value) for value in features_data_slice.values.flatten()
        ]:
            features_individual.append(list(features_data_slice[ticker_label].values))
            features_sector.append(list(features_data_slice.values.flatten()))
            label.append(data[ticker_label].iloc[i + features_length])
    return pd.DataFrame(
        data={"features_individual": features_individual, "features_sector": features_sector, "label": label}
    )
