"""Methods to build labeled data from a DataFrame for time series forecasting."""

import math
from typing import List

import pandas as pd

from src.tools.constants import PriceAttribute
from src.tools.helper_methods import consecutive_timestamps
from src.tools.labeled_data_builder.balance_data import undersample


def create_labeled_data(
    attribute_label: PriceAttribute,
    ticker_label: str,
    tickers_features: List[str],
    data: pd.DataFrame,
    features_length: int,
) -> pd.DataFrame:
    """Create labeled data for time series forecasting, using given historical data for a ticker, using features_length
    previous values in the time series as features_individual and features_sector and the next value as label. We build
    features for both the individual and sector approach, so that both approaches use the same data points, to allow for
    fairer comparison. The data is then balanced using under-sampling.

    Args:
        attribute_label (PriceAttribute): The price attribute we want to predict for (the label).
        ticker_label (str): The code for the asset we want to predict for (the label).
        tickers_features (List[str]): The codes for the assets we want to use as features_sector for the prediction.
        data (pd.DataFrame): The historical time series for the tickers.
        features_length (int): The number of previous rows to use as features_sector to predict the next one (label).

    Returns:
        labeled_data (pd.DataFrame): The created labeled data, contains a columns for features_individual,
            features_sector, label and true_return.

    """

    if (ticker_label, attribute_label.value) not in data.columns:
        raise ValueError(
            "Parameters 'ticker_label' and 'attribute_label' must represent a valid column in the 'data' "
            "provided as parameter."
        )

    if not tickers_features or not {column[0] for column in data.columns}.issuperset(set(tickers_features)):
        raise ValueError(
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter."
        )

    if features_length < 1:
        raise ValueError("Parameter 'features_length' must be a strictly positive integer (>= 1).")

    if ticker_label not in tickers_features:
        tickers_features.append(ticker_label)

    timestamp = []
    features_individual = []
    features_sector = []
    label = []
    true_return = []
    for i in range(len(data) - features_length):
        features_data_slice = data[tickers_features].iloc[i : i + features_length]
        if (
            not math.isnan(data[(ticker_label, attribute_label.value)].iloc[i + features_length])
            and True not in [math.isnan(value) for value in features_data_slice.values.flatten()]
            and consecutive_timestamps(data[tickers_features].iloc[i : i + features_length + 1].index)
        ):
            timestamp.append(data.index[i + features_length])
            features_individual.append(list(features_data_slice[ticker_label].values.flatten()))
            features_sector.append(list(features_data_slice.values.flatten()))
            label.append(data[(ticker_label, attribute_label.value)].iloc[i + features_length] > 0)
            true_return.append(data[(ticker_label, attribute_label.value)].iloc[i + features_length])
    labeled_data = pd.DataFrame(
        data={
            "timestamp": timestamp,
            "features_individual": features_individual,
            "features_sector": features_sector,
            "label_classification": label,
            "label_regression": true_return,
        }
    ).set_index("timestamp")
    labeled_data.index = pd.DatetimeIndex(labeled_data.index)

    if attribute_label == PriceAttribute.CLOSE:
        return undersample(labeled_data=labeled_data)
    return labeled_data
