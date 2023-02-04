"""Common helper methods for other classes."""

import math
from typing import Union

import pandas as pd

from src.tools.constants import PriceAttribute


def extract_changes_from_dataframe(attribute: PriceAttribute, data: pd.DataFrame) -> pd.Series:
    """Create a pandas Series representing changes (as a percentage) for each row in the given pandas DataFrame.

    Args:
        attribute (PriceAttribute): The price attribute (column) to extract changes for.
        data (pd.DataFrame): The pandas DataFrame containing columns 'Open' and 'Close' columns.

    Returns:
        change_series (pd.Series): The changes (continuous values), extracted from the DataFrame.

    """

    if data.empty:
        change_series = pd.Series(dtype=object)
        change_series.index = pd.DatetimeIndex(change_series.index, name="Date")
        return change_series

    if attribute == PriceAttribute.OPEN or attribute == PriceAttribute.VOLUME:
        raise ValueError("Cannot extract changes for price attributes 'Open' or 'Volume'.")

    def row_change_value(row: pd.Series) -> Union[bool, float]:
        if math.isnan(row["Open"]) or math.isnan(row[attribute.value]):
            return math.nan
        return (row[attribute.value] - row["Open"]) / row["Open"]

    return data.apply(lambda row: row_change_value(row), axis=1)
