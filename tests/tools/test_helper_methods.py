"""Tests for methods in file balance_data.py."""

import math
from unittest import TestCase

import pandas as pd

from src.tools.constants import PriceAttribute
from src.tools.helper_methods import consecutive_timestamps, extract_changes_from_dataframe


class TestHelperMethods(TestCase):
    """Test class for methods in file balance_data.py."""

    # Tests for method extract_changes_from_dataframe()

    def test_extract_changes_from_dataframe_no_nan(self):

        # Arrange
        attribute = PriceAttribute.CLOSE
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
        changes_series = extract_changes_from_dataframe(attribute=attribute, data=data)

        # Assert
        expected_changes_series = pd.Series(data={"2022-11-03": -1 / 89, "2022-11-04": 4 / 88, "2022-11-07": 0.0})
        expected_changes_series.index = pd.DatetimeIndex(expected_changes_series.index, name="Date")
        self.assertTrue(expected_changes_series.equals(changes_series))

    def test_extract_changes_from_dataframe_open_nan(self):

        # Arrange
        attribute = PriceAttribute.CLOSE
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
        changes_series = extract_changes_from_dataframe(attribute=attribute, data=data)

        # Assert
        expected_changes_series = pd.Series(
            data={
                "2022-11-03": -1 / 89,
                "2022-11-04": math.nan,
                "2022-11-07": 0.0,
            }
        )
        expected_changes_series.index = pd.DatetimeIndex(expected_changes_series.index, name="Date")
        self.assertTrue(expected_changes_series.equals(changes_series))

    def test_extract_changes_from_dataframe_close_nan(self):

        # Arrange
        attribute = PriceAttribute.CLOSE
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
        changes_series = extract_changes_from_dataframe(attribute=attribute, data=data)

        # Assert
        expected_changes_series = pd.Series(
            data={
                "2022-11-03": -1 / 89,
                "2022-11-04": 4 / 88,
                "2022-11-07": math.nan,
            }
        )
        expected_changes_series.index = pd.DatetimeIndex(expected_changes_series.index, name="Date")
        self.assertTrue(expected_changes_series.equals(changes_series))

    def test_extract_changes_from_dataframe_empty_data(self):

        # Arrange
        attribute = PriceAttribute.CLOSE
        data = pd.DataFrame(columns=["Date", "Open", "Close"])
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act
        changes_series = extract_changes_from_dataframe(attribute=attribute, data=data)

        # Assert
        expected_changes_series = pd.Series(dtype=object)
        expected_changes_series.index = pd.DatetimeIndex(expected_changes_series.index, name="Date")
        self.assertTrue(expected_changes_series.equals(changes_series))

    def test_extract_changes_from_dataframe_attribute_is_open(self):

        # Arrange
        attribute = PriceAttribute.OPEN
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-03", "2022-11-04", "2022-11-07"],
                "Open": [89, 88, 91],
                "Close": [88, 92, 91],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            extract_changes_from_dataframe(attribute=attribute, data=data)
        self.assertEqual("Cannot extract changes for price attributes 'Open' or 'Volume'.", str(e.exception))

    def test_extract_changes_from_dataframe_attribute_is_volume(self):

        # Arrange
        attribute = PriceAttribute.VOLUME
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-03", "2022-11-04", "2022-11-07"],
                "Open": [89, 88, 91],
                "Close": [88, 92, 91],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            extract_changes_from_dataframe(attribute=attribute, data=data)
        self.assertEqual("Cannot extract changes for price attributes 'Open' or 'Volume'.", str(e.exception))

    # Tests for method consecutive_timestamps()

    def test_consecutive_timestamps_empty_timestamps_list(self):

        # Arrange
        timestamps = []

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertTrue(timestamps_are_consecutive)

    def test_consecutive_timestamps_timestamps_list_contains_one_element(self):

        # Arrange
        timestamps = [pd.Timestamp("2022-11-07 10:00")]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertTrue(timestamps_are_consecutive)

    def test_consecutive_timestamps_consecutive(self):

        # Arrange
        timestamps = [
            pd.Timestamp("2022-11-07 10:00"),
            pd.Timestamp("2022-11-07 11:00"),
            pd.Timestamp("2022-11-07 12:00"),
            pd.Timestamp("2022-11-07 13:00"),
            pd.Timestamp("2022-11-07 14:00"),
            pd.Timestamp("2022-11-07 15:00"),
        ]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertTrue(timestamps_are_consecutive)

    def test_consecutive_timestamps_consecutive_but_reverse_order(self):

        # Arrange
        timestamps = [
            pd.Timestamp("2022-11-07 15:00"),
            pd.Timestamp("2022-11-07 14:00"),
            pd.Timestamp("2022-11-07 13:00"),
            pd.Timestamp("2022-11-07 12:00"),
            pd.Timestamp("2022-11-07 11:00"),
            pd.Timestamp("2022-11-07 10:00"),
        ]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertFalse(timestamps_are_consecutive)

    def test_consecutive_timestamps_first_timestamp_not_consecutive(self):

        # Arrange
        timestamps = [
            pd.Timestamp("2022-11-07 09:00"),
            pd.Timestamp("2022-11-07 11:00"),
            pd.Timestamp("2022-11-07 12:00"),
            pd.Timestamp("2022-11-07 13:00"),
            pd.Timestamp("2022-11-07 14:00"),
            pd.Timestamp("2022-11-07 15:00"),
        ]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertFalse(timestamps_are_consecutive)

    def test_consecutive_timestamps_last_timestamp_not_consecutive(self):

        # Arrange
        timestamps = [
            pd.Timestamp("2022-11-07 10:00"),
            pd.Timestamp("2022-11-07 11:00"),
            pd.Timestamp("2022-11-07 12:00"),
            pd.Timestamp("2022-11-07 13:00"),
            pd.Timestamp("2022-11-07 14:00"),
            pd.Timestamp("2022-11-07 16:00"),
        ]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertFalse(timestamps_are_consecutive)

    def test_consecutive_timestamps_internal_break_in_consecutiveness(self):

        # Arrange
        timestamps = [
            pd.Timestamp("2022-11-07 10:00"),
            pd.Timestamp("2022-11-07 11:00"),
            pd.Timestamp("2022-11-07 12:00"),
            pd.Timestamp("2022-11-07 14:00"),
            pd.Timestamp("2022-11-07 15:00"),
            pd.Timestamp("2022-11-07 16:00"),
        ]

        # Act
        timestamps_are_consecutive = consecutive_timestamps(timestamps=timestamps)

        # Assert
        self.assertFalse(timestamps_are_consecutive)
