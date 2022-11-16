"""Common helper methods for other classes."""

import math
from typing import Union

import pandas as pd


def extract_returns_from_dataframe(data: pd.DataFrame) -> pd.Series:
    """Create a pandas Series representing returns for each row in the given pandas DataFrame.

    Args:
        data (pd.DataFrame): The pandas DataFrame containing columns 'Open' and 'Close' columns.

    Returns:
        returns_series (pd.Series): The returns, extracted from the DataFrame.

    """

    if data.empty:
        returns_series = pd.Series(dtype=object)
        returns_series.index = pd.DatetimeIndex(returns_series.index, name="Date")
        return returns_series

    def daily_return_value(row: pd.Series) -> Union[bool, float]:
        if math.isnan(row["Open"]) or math.isnan(row["Close"]):
            return math.nan
        return row["Close"] > row["Open"]

    return data.apply(lambda row: daily_return_value(row), axis=1)
