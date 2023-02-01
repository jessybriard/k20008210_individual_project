"""Tests for methods in file balance_data.py."""

import math
from unittest import TestCase

import pandas as pd

from src.tools.helper_methods import extract_returns_from_dataframe


class TestHelperMethods(TestCase):
    """Test class for methods in file balance_data.py."""

    # Tests for method extract_returns_from_dataframe()

    def test_extract_returns_from_dataframe_no_nan(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-03", "2022-11-04", "2022-11-07"],
                "Open": [89, 88, 91],
                "Close": [88, 92, 91],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act
        returns_series = extract_returns_from_dataframe(data=data)

        # Assert
        expected_returns_series = pd.Series(data={"2022-11-03": -1 / 89, "2022-11-04": 4 / 88, "2022-11-07": 0.0})
        expected_returns_series.index = pd.DatetimeIndex(expected_returns_series.index, name="Date")
        self.assertTrue(expected_returns_series.equals(returns_series))

    def test_extract_returns_from_dataframe_open_nan(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-03", "2022-11-04", "2022-11-07"],
                "Open": [89, math.nan, 91],
                "Close": [88, 92, 91],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act
        returns_series = extract_returns_from_dataframe(data=data)

        # Assert
        expected_returns_series = pd.Series(
            data={
                "2022-11-03": -1 / 89,
                "2022-11-04": math.nan,
                "2022-11-07": 0.0,
            }
        )
        expected_returns_series.index = pd.DatetimeIndex(expected_returns_series.index, name="Date")
        self.assertTrue(expected_returns_series.equals(returns_series))

    def test_extract_returns_from_dataframe_close_nan(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-03", "2022-11-04", "2022-11-07"],
                "Open": [89, 88, 91],
                "Close": [88, 92, math.nan],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act
        returns_series = extract_returns_from_dataframe(data=data)

        # Assert
        expected_returns_series = pd.Series(
            data={
                "2022-11-03": -1 / 89,
                "2022-11-04": 4 / 88,
                "2022-11-07": math.nan,
            }
        )
        expected_returns_series.index = pd.DatetimeIndex(expected_returns_series.index, name="Date")
        self.assertTrue(expected_returns_series.equals(returns_series))

    def test_extract_returns_from_dataframe_empty_data(self):

        # Arrange
        data = pd.DataFrame(columns=["Date", "Open", "Close"])
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act
        returns_series = extract_returns_from_dataframe(data=data)

        # Assert
        expected_returns_series = pd.Series(dtype=object)
        expected_returns_series.index = pd.DatetimeIndex(expected_returns_series.index, name="Date")
        self.assertTrue(expected_returns_series.equals(returns_series))
