"""Tests for methods in labeled_data_builder/time_series_forecasting.py."""

import math
from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.time_series_forecasting import (
    create_labeled_data_individual_approach,
    create_labeled_data_sector_approach,
)


class TestLabeledDataBuilderTimeSeriesForecasting(TestCase):
    """Test class for methods in labeled_data_builder/time_series_forecasting.py."""

    # Tests for method create_labeled_data_individual_approach()

    def test_create_labeled_data_individual_approach_single_ticker_data(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "features": [[True, False], [False, False], [False, True], [True, True]],
                "label": [False, True, True, False],
            }
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_multiple_tickers_data(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 3

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "features": [[True, False, False], [False, False, True], [False, True, True]],
                "label": [True, True, False],
            }
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_data_contains_nan(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, math.nan, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [[True, True]], "label": [False]})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_data_empty(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(data={"Date": [], "CL=F": [], "EUR=X": []})
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_data_nan_only(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_ticker_not_in_data_columns(self):

        # Arrange
        ticker = "EUR=X"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_individual_approach(ticker=ticker, data=data, features_length=features_length)
        self.assertEqual(
            str(e.exception), "Parameter 'ticker' must represent a valid column in the 'data' provided as parameter."
        )

    def test_create_labeled_data_individual_approach_ticker_empty_str(self):

        # Arrange
        ticker = ""
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_individual_approach(ticker=ticker, data=data, features_length=features_length)
        self.assertEqual(
            str(e.exception), "Parameter 'ticker' must represent a valid column in the 'data' provided as parameter."
        )

    def test_create_labeled_data_individual_approach_features_length_zero(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 0

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_individual_approach(ticker=ticker, data=data, features_length=features_length)
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_individual_approach_features_length_negative(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = -1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_individual_approach(ticker=ticker, data=data, features_length=features_length)
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_individual_approach_one_example(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 5

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [[True, False, False, True, True]], "label": [False]})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_individual_approach_zero_example(self):

        # Arrange
        ticker = "CL=F"
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        data = data.set_index("Date")
        data.index = pd.DatetimeIndex(data.index)
        features_length = 6

        # Act
        labeled_data = create_labeled_data_individual_approach(
            ticker=ticker, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    # Tests for method create_labeled_data_sector_approach()

    def test_create_labeled_data_sector_approach_ticker_label_empty_str(self):

        # Arrange
        ticker_label = ""
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'ticker_label' must represent a valid column in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_sector_approach_ticker_label_not_in_data_columns(self):

        # Arrange
        ticker_label = "GBPUSD=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'ticker_label' must represent a valid column in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_sector_approach_tickers_features_empty_list(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = []
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_sector_approach_tickers_features_not_subset_of_data_columns(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "GC=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_sector_approach_features_length_zero(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, False, False, False, False, False],
            }
        )
        features_length = 0

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_sector_approach_features_length_negative(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = -1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data_sector_approach(
                ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_sector_approach_nan_in_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, math.nan, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={"features": [[False, False, True, True], [True, True, True, False]], "label": [False, True]}
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_nan_in_label(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [math.nan, True, False, math.nan, False, True],
            }
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={"features": [[True, False], [False, True], [True, True]], "label": [False, False, True]}
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_data_empty(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": [],
                "CL=F": [],
                "EUR=X": [],
            }
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_data_nan_only(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                "EUR=X": [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            }
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_one_example(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 5

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "features": [
                    [True, False, False, True, False, False, True, True, True, False],
                ],
                "label": [True],
            }
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_zero_example(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 6

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(data={"features": [], "label": []})
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_ticker_label_in_tickers_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "features": [
                    [True, False, False, True, False, False],
                    [False, True, False, False, True, True],
                    [False, False, True, True, True, False],
                ],
                "label": [True, False, True],
            }
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_sector_approach_ticker_label_not_in_tickers_features(self):

        # Arrange
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                "Date": ["2022-11-07", "2022-11-08", "2022-11-09", "2022-11-10", "2022-11-11", "2022-11-14"],
                "CL=F": [True, False, False, True, True, False],
                "EUR=X": [False, True, False, True, False, True],
            }
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data_sector_approach(
            ticker_label=ticker_label, tickers_features=tickers_features, data=data, features_length=features_length
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "features": [[True, False, False], [False, False, True], [False, True, True]],
                "label": [True, False, True],
            }
        )
        self.assertTrue(expected_labeled_data.equals(labeled_data))
