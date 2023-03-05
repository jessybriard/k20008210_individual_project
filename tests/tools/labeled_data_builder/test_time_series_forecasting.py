"""Tests for methods in labeled_data_builder/time_series_forecasting.py."""

import math
from unittest import TestCase

import pandas as pd

from src.tools.constants import PriceAttribute
from src.tools.labeled_data_builder.time_series_forecasting import create_labeled_data


class TestLabeledDataBuilderTimeSeriesForecasting(TestCase):
    """Test class for methods in labeled_data_builder/time_series_forecasting.py."""

    # Tests for method create_labeled_data()

    def test_create_labeled_data_ticker_label_empty_str(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = ""
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'ticker_label' and 'attribute_label' must represent a valid column in the 'data' provided as "
            "parameter.",
        )

    def test_create_labeled_data_ticker_label_not_in_data_columns(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "GBPUSD=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'ticker_label' and 'attribute_label' must represent a valid column in the 'data' provided as "
            "parameter.",
        )

    def test_create_labeled_data_attribute_label_not_in_data_columns(self):

        # Arrange
        attribute_label = PriceAttribute.HIGH
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'ticker_label' and 'attribute_label' must represent a valid column in the 'data' provided as "
            "parameter.",
        )

    def test_create_labeled_data_ticker_label_and_attribute_label_not_in_data_columns(self):

        # Arrange
        attribute_label = PriceAttribute.HIGH
        ticker_label = "GBPUSD=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'ticker_label' and 'attribute_label' must represent a valid column in the 'data' provided as "
            "parameter.",
        )

    def test_create_labeled_data_tickers_features_empty_list(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = []
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_tickers_features_not_subset_of_data_columns(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "GC=F"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(
            str(e.exception),
            "Parameter 'tickers_features' must represent valid columns in the 'data' provided as parameter.",
        )

    def test_create_labeled_data_features_length_zero(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, False, False, False, False, False],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 0

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_features_length_negative(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [True, False, False, True, True, False],
                ("EUR=X", "Close"): [False, True, False, True, False, True],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = -1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            create_labeled_data(
                attribute_label=attribute_label,
                ticker_label=ticker_label,
                tickers_features=tickers_features,
                data=data,
                features_length=features_length,
            )
        self.assertEqual(str(e.exception), "Parameter 'features_length' must be a strictly positive integer (>= 1).")

    def test_create_labeled_data_nan_in_features(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, math.nan, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[-0.04, 0.22], [0.22, -0.12]],
                "features_sector": [[-0.05, -0.04, 0.05, 0.22], [0.05, 0.22, 0.2, -0.12]],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_nan_in_label(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, math.nan, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 12:00"],
                "features_individual": [[-0.1, 0.08]],
                "features_sector": [[0.1, -0.1, -0.06, 0.08]],
                "label_classification": [False],
                "label_regression": [-0.04],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_data_empty(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [],
                ("EUR=X", "Close"): [],
            }
        )
        data.index = pd.DatetimeIndex(pd.Series(data=[], name="Date", dtype="str"))
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": [],
                "features_individual": [],
                "features_sector": [],
                "label_classification": [],
                "label_regression": [],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_data_nan_only(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
                ("EUR=X", "Close"): [math.nan, math.nan, math.nan, math.nan, math.nan, math.nan],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 2

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": [],
                "features_individual": [],
                "features_sector": [],
                "label_classification": [],
                "label_regression": [],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_one_example(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 5

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 15:00"],
                "features_individual": [[-0.1, 0.08, -0.04, 0.22, -0.12]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label_classification": [True],
                "label_regression": [0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_zero_example(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 6

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": [],
                "features_individual": [],
                "features_sector": [],
                "label_classification": [],
                "label_regression": [],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_ticker_label_in_tickers_features(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00"],
                "features_individual": [[-0.1, 0.08, -0.04], [0.08, -0.04, 0.22]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04],
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                ],
                "label_classification": [True, False],
                "label_regression": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[0.08, -0.04, 0.22], [-0.04, 0.22, -0.12]],
                "features_sector": [
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                    [-0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))

    def test_create_labeled_data_ticker_label_not_in_tickers_features(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00"],
                "features_individual": [[-0.1, 0.08, -0.04], [0.08, -0.04, 0.22]],
                "features_sector": [
                    [0.1, -0.1, -0.06, 0.08, -0.05, -0.04],
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                ],
                "label_classification": [True, False],
                "label_regression": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[0.08, -0.04, 0.22], [-0.04, 0.22, -0.12]],
                "features_sector": [
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                    [-0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))

    def test_create_labeled_data_multiple_price_attributes_in_data(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("CL=F", "High"): [0.2, 0.12, 0.1, 0.1, 0.4, 0.2],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
                ("EUR=X", "High"): [0.2, 0.16, 0.08, 0.44, 0.24, 0.1],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00"],
                "features_individual": [[-0.1, 0.2, 0.08, 0.16, -0.04, 0.08], [0.08, 0.16, -0.04, 0.08, 0.22, 0.44]],
                "features_sector": [
                    [0.1, 0.2, -0.1, 0.2, -0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08],
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                ],
                "label_classification": [True, False],
                "label_regression": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[0.08, 0.16, -0.04, 0.08, 0.22, 0.44], [-0.04, 0.08, 0.22, 0.44, -0.12, 0.24]],
                "features_sector": [
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                    [-0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44, 0.2, 0.4, -0.12, 0.24],
                ],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))

    def test_create_labeled_data_close_attribute_undersamples_if_unbalanced_data(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("CL=F", "High"): [0.2, 0.12, 0.1, 0.1, 0.4, 0.2],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
                ("EUR=X", "High"): [0.2, 0.16, 0.08, 0.44, 0.24, 0.1],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data_1 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00"],
                "features_individual": [[-0.1, 0.2, 0.08, 0.16, -0.04, 0.08], [0.08, 0.16, -0.04, 0.08, 0.22, 0.44]],
                "features_sector": [
                    [0.1, 0.2, -0.1, 0.2, -0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08],
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                ],
                "label_classification": [True, False],
                "label_regression": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data_1.index = pd.DatetimeIndex(expected_labeled_data_1.index)
        expected_labeled_data_2 = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[0.08, 0.16, -0.04, 0.08, 0.22, 0.44], [-0.04, 0.08, 0.22, 0.44, -0.12, 0.24]],
                "features_sector": [
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                    [-0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44, 0.2, 0.4, -0.12, 0.24],
                ],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data_2.index = pd.DatetimeIndex(expected_labeled_data_2.index)
        self.assertTrue(expected_labeled_data_1.equals(labeled_data) or expected_labeled_data_2.equals(labeled_data))

    def test_create_labeled_data_close_attribute_does_not_undersample_if_balanced_data(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2],
                ("CL=F", "High"): [0.2, 0.12, 0.1, 0.1, 0.4],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12],
                ("EUR=X", "High"): [0.2, 0.16, 0.08, 0.44, 0.24],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00"],
                "features_individual": [[-0.1, 0.2, 0.08, 0.16, -0.04, 0.08], [0.08, 0.16, -0.04, 0.08, 0.22, 0.44]],
                "features_sector": [
                    [0.1, 0.2, -0.1, 0.2, -0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08],
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                ],
                "label_classification": [True, False],
                "label_regression": [0.22, -0.12],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_high_attribute_does_not_undersample(self):

        # Arrange
        attribute_label = PriceAttribute.HIGH
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("CL=F", "High"): [0.2, 0.12, 0.1, 0.1, 0.4, 0.2],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
                ("EUR=X", "High"): [0.2, 0.16, 0.08, 0.44, 0.0, 0.1],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [
                    [-0.1, 0.2, 0.08, 0.16, -0.04, 0.08],
                    [0.08, 0.16, -0.04, 0.08, 0.22, 0.44],
                    [-0.04, 0.08, 0.22, 0.44, -0.12, 0.0],
                ],
                "features_sector": [
                    [0.1, 0.2, -0.1, 0.2, -0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08],
                    [-0.06, 0.12, 0.08, 0.16, -0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44],
                    [-0.05, 0.1, -0.04, 0.08, 0.05, 0.1, 0.22, 0.44, 0.2, 0.4, -0.12, 0.0],
                ],
                "label_classification": [True, False, True],
                "label_regression": [0.44, 0.0, 0.1],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_low_attribute_does_not_undersample(self):

        # Arrange
        attribute_label = PriceAttribute.LOW
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("CL=F", "Low"): [-0.2, -0.12, -0.1, -0.1, -0.4, -0.2],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
                ("EUR=X", "Low"): [-0.2, -0.16, -0.08, -0.44, 0.0, -0.1],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 10:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 13:00", "2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [
                    [-0.1, -0.2, 0.08, -0.16, -0.04, -0.08],
                    [0.08, -0.16, -0.04, -0.08, 0.22, -0.44],
                    [-0.04, -0.08, 0.22, -0.44, -0.12, 0.0],
                ],
                "features_sector": [
                    [0.1, -0.2, -0.1, -0.2, -0.06, -0.12, 0.08, -0.16, -0.05, -0.1, -0.04, -0.08],
                    [-0.06, -0.12, 0.08, -0.16, -0.05, -0.1, -0.04, -0.08, 0.05, -0.1, 0.22, -0.44],
                    [-0.05, -0.1, -0.04, -0.08, 0.05, -0.1, 0.22, -0.44, 0.2, -0.4, -0.12, 0.0],
                ],
                "label_classification": [False, False, False],
                "label_regression": [-0.44, 0.0, -0.1],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data))

    def test_create_labeled_data_contains_break_in_consecutiveness(self):

        # Arrange
        attribute_label = PriceAttribute.CLOSE
        ticker_label = "EUR=X"
        tickers_features = ["CL=F", "EUR=X"]
        data = pd.DataFrame(
            data={
                ("CL=F", "Close"): [0.1, -0.06, -0.05, 0.05, 0.2, -0.1],
                ("EUR=X", "Close"): [-0.1, 0.08, -0.04, 0.22, -0.12, 0.05],
            }
        )
        data.index = pd.DatetimeIndex(
            pd.Series(
                data=[
                    "2022-11-07 09:00",
                    "2022-11-07 11:00",
                    "2022-11-07 12:00",
                    "2022-11-07 13:00",
                    "2022-11-07 14:00",
                    "2022-11-07 15:00",
                ],
                name="Date",
            )
        )
        features_length = 3

        # Act
        labeled_data = create_labeled_data(
            attribute_label=attribute_label,
            ticker_label=ticker_label,
            tickers_features=tickers_features,
            data=data,
            features_length=features_length,
        )

        # Assert
        expected_labeled_data = pd.DataFrame(
            data={
                "timestamp": ["2022-11-07 14:00", "2022-11-07 15:00"],
                "features_individual": [[0.08, -0.04, 0.22], [-0.04, 0.22, -0.12]],
                "features_sector": [
                    [-0.06, 0.08, -0.05, -0.04, 0.05, 0.22],
                    [-0.05, -0.04, 0.05, 0.22, 0.2, -0.12],
                ],
                "label_classification": [False, True],
                "label_regression": [-0.12, 0.05],
            }
        ).set_index("timestamp")
        expected_labeled_data.index = pd.DatetimeIndex(expected_labeled_data.index)
        self.assertTrue(expected_labeled_data.equals(labeled_data) or expected_labeled_data.equals(labeled_data))
