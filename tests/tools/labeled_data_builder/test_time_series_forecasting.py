"""Tests for methods in labeled_data_builder/time_series_forecasting.py."""

import math
from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.time_series_forecasting import create_labeled_data


class TestLabeledDataBuilderTimeSeriesForecasting(TestCase):
    """Test class for methods in labeled_data_builder/time_series_forecasting.py."""

    # Tests for method create_labeled_data()

    def test_create_labeled_data_ticker_label_empty_str(self):

        # Arrange
        ticker_label = ""
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'ticker_label' must represent a valid column in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_ticker_label_not_in_data_columns(self):

        # Arrange
        ticker_label = "GBPUSD=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'ticker_label' must represent a valid column in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_tickers_features_empty_list(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = []
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_tickers_features_not_subset_of_data_columns(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "GC=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_features_length_zero(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 0

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_features_length_negative(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = -1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_nan_in_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, math.nan, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-11", "2022-11-14"],
                "features_individual": [[-0.04, 0.22], [0.22, -0.12]],
                "features_sector": [[-0.05, -0.04, 0.05, 0.22], [0.05, 0.22, 0.2, -0.12]],
                "label": [False, True],
                "true_return": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_nan_in_label(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, math.nan, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-09"],
                "features_individual": [[-0.1, 0.08]],
                "features_sector": [[0.1, -0.1, -0.06, 0.08]],
                "label": [False],
                "true_return": [-0.04],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_data_empty(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": [],
                "CL=F": [],
                "EUR=X": [],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={"timestamp": [], "features_individual": [], "features_sector": [], "label": [], "true_return": []}
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_data_nan_only(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                "EUR=X": [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={"timestamp": [], "features_individual": [], "features_sector": [], "label": [], "true_return": []}
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_one_example(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 5

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-14"],
                "features_individual": [[-0.1, 0.08, -0.04, 0.22, -0.12]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label": [True],
                "true_return": [0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_zero_example(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 6

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={"timestamp": [], "features_individual": [], "features_sector": [], "label": [], "true_return": []}
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_ticker_label_in_tickers_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-10", "2022-11-11"],
                "features_individual": [[-0.1, 0.08, -0.04], [0.08, -0.04, 0.22]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04],
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                ],
                "label": [True, False],
                "true_return": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-11", "2022-11-14"],
                "features_individual": [[0.08, -0.04, 0.22], [-0.04, 0.22, -0.12]],
                "features_sector": [
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                    [-0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label": [False, True],
                "true_return": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))

    def test_create_labeled_data_ticker_label_not_in_tickers_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                "EUR=X": [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        ).set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-10", "2022-11-11"],
                "features_individual": [[-0.1, 0.08, -0.04], [0.08, -0.04, 0.22]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04],
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                ],
                "label": [True, False],
                "true_return": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-11", "2022-11-14"],
                "features_individual": [[0.08, -0.04, 0.22], [-0.04, 0.22, -0.12]],
                "features_sector": [
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                    [-0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label": [False, True],
                "true_return": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))
