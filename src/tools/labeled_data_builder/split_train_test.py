"""Method(s) to split labeled data into train and test datasets."""

from typing import Tuple

import pandas as pd


def split_time_series_forecasting(
    labeled_data: pd.DataFrame, train_percentage: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled data, for time series forecasting, into train and test datasets. To more accurately represent
    the production scenario of time series forecasting, we do not sample training data randomly, but split the data
    around a fixed point in time (we train models on past data, and predict/test on future data).

    Args:
        labeled_data (pd.DataFrame): The labeled data, a pandas DataFrame with columns 'features' and 'label'.
        train_percentage (float): Proportion of labeled data going into the train dataset, between 0 and 1 inclusive.

    Returns:
        train_data (pd.DataFrame): The training data, starting subset of the labeled_data.
        test_data (pd.DataFrame): The testing data, ending subset of the labeled_data

    """

    if not list(labeled_data.columns) == ["features", "label"]:
        raise ValueError(
            "Invalid 'labeled_data' provided, the DataFrame must be composed of the columns 'features' and 'labels'."
        )

    if not 0 <= train_percentage <= 1:
        raise ValueError("Parameter 'train_percentage' must be a number between 0 and 1 inclusive.")

    train_data_len = int(len(labeled_data) * train_percentage)
    train_data = labeled_data.iloc[:train_data_len]
    test_data = labeled_data.iloc[train_data_len:].reset_index(drop=True)

    return train_data, test_data
