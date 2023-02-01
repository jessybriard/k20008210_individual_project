"""Tests for methods in labeled_data_builder/split_train_test.py."""

from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.split_train_test import split_time_series_forecasting


class TestLabeledDataBuilderSplitTrainTest(TestCase):
    """Test class for methods in labeled_data_builder/split_train_test.py."""

    # Tests for method split_time_series_forecasting()

    def test_split_time_series_forecasting_labeled_data_no_row(self):

        # Arrange
        data = pd.DataFrame(columns=["features", "labels"])
        train_percentage = 0.8

        # Act
        train_data, test_data = split_time_series_forecasting(data=data, train_percentage=train_percentage)

        # Assert
        expected_train_data = pd.DataFrame(columns=["features", "labels"])
        self.assertTrue(expected_train_data.equals(train_data))
        expected_test_data = pd.DataFrame(columns=["features", "labels"])
        self.assertTrue(expected_test_data.equals(test_data))

    def test_split_time_series_forecasting_train_percentage_between_zero_and_one(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        train_percentage = 0.8

        # Act
        train_data, test_data = split_time_series_forecasting(data=data, train_percentage=train_percentage)

        # Assert
        expected_train_data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True]],
                "label": [False, True, True],
            }
        )
        self.assertTrue(expected_train_data.equals(train_data))
        expected_test_data = pd.DataFrame(
            data={
                "features": [[True, True]],
                "label": [False],
            }
        )
        self.assertTrue(expected_test_data.equals(test_data.reset_index(drop=True)))

    def test_split_time_series_forecasting_train_percentage_zero(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        train_percentage = 0

        # Act
        train_data, test_data = split_time_series_forecasting(data=data, train_percentage=train_percentage)

        # Assert
        self.assertTrue(train_data.empty)
        self.assertEqual(["features", "label"], list(train_data.columns))
        self.assertTrue(data.equals(test_data))

    def test_split_time_series_forecasting_train_percentage_one(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        train_percentage = 1

        # Act
        train_data, test_data = split_time_series_forecasting(data=data, train_percentage=train_percentage)

        # Assert
        self.assertTrue(data.equals(train_data))
        self.assertTrue(test_data.empty)
        self.assertEqual(["features", "label"], list(test_data.columns))

    def test_split_time_series_forecasting_train_percentage_under_zero(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        train_percentage = -0.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            split_time_series_forecasting(data=data, train_percentage=train_percentage)
        self.assertEqual(str(e.exception), "Parameter 'train_percentage' must be a number between 0 and 1 inclusive.")

    def test_split_time_series_forecasting_train_percentage_over_one(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        train_percentage = 1.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            split_time_series_forecasting(data=data, train_percentage=train_percentage)
        self.assertEqual(str(e.exception), "Parameter 'train_percentage' must be a number between 0 and 1 inclusive.")

    def test_split_time_series_forecasting_non_arbitrary_index(self):

        # Arrange
        open_series = pd.Series(
            data={
                "2022-11-07": 91.00,
                "2022-11-08": 91.87,
                "2022-11-09": 88.57,
                "2022-11-10": 85.85,
                "2022-11-11": 86.27,
                "2022-11-14": 89.02,
            }
        )
        open_series.index = pd.DatetimeIndex(open_series.index)
        close_series = pd.Series(
            data={
                "2022-11-07": 91.79,
                "2022-11-08": 88.91,
                "2022-11-09": 85.83,
                "2022-11-10": 86.47,
                "2022-11-11": 88.96,
                "2022-11-14": 88.16,
            }
        )
        close_series.index = pd.DatetimeIndex(close_series.index)
        data = pd.DataFrame(data={"Open": open_series, "Close": close_series})
        train_percentage = 0.8

        # Act
        train_data, test_data = split_time_series_forecasting(data=data, train_percentage=train_percentage)

        # Assert
        expected_train_data = pd.DataFrame(data={"Open": open_series.iloc[:4], "Close": close_series.iloc[:4]})
        self.assertTrue(expected_train_data.equals(train_data))
        self.assertTrue(
            pd.DatetimeIndex(
                ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10"], dtype="datetime64[ns]", name="Date", freq=None
            ).equals(train_data.index)
        )
        expected_test_data = pd.DataFrame(data={"Open": open_series.iloc[4:], "Close": close_series.iloc[4:]})
        self.assertTrue(expected_test_data.equals(test_data))
        self.assertTrue(
            pd.DatetimeIndex(["2022-11-11", "2022-11-14"], dtype="datetime64[ns]", name="Date", freq=None).equals(
                test_data.index
            )
        )
