"""Tests for methods in file correlation_analysis.py."""

import math
import os
import pickle
from unittest import TestCase
from unittest.mock import patch

from scipy.stats import pearsonr

from src.tools.constants import YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.correlation_analysis import (
    correlation_analysis_lists_cardinal_product,
    correlation_analysis_single_combination,
)


class TestCorrelationAnalysis(TestCase):
    """Test class for methods in correlation_analysis.py."""

    def setUp(self) -> None:
        self.TEST_DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/test_data"

    def mock_get_data_side_effect(self, **kwargs):
        self.get_data_parameters = kwargs
        return self.data

    def mock_pearsonr_side_effect(self, list1, list2):
        self.pearsonr_parameters = (list1, list2)
        return pearsonr(list1, list2)

    # Tests for method correlation_analysis_single_combination()

    def load_data_single_combination(self):
        with open(f"{self.TEST_DATA_DIR}/yf_download_multiple_tickers_output.pickle", "rb") as file:
            self.data = pickle.load(file)

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_ticker1_empty_str(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = ""
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(str(e.exception), "Parameters 'ticker1' and 'ticker2' must be non-empty.")

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_ticker2_empty_str(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = ""
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(str(e.exception), "Parameters 'ticker1' and 'ticker2' must be non-empty.")

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_ticker1_equals_ticker2(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "CL=F"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(str(e.exception), "Parameters 'ticker1' and 'ticker2' must represent different assets.")

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker1_empty_str(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = ""
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker2_empty_str(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = ""
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker1_default(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertEqual(
            [
                91.79000091552734,
                88.91000366210938,
                85.83000183105469,
                86.47000122070312,
                88.95999908447266,
                88.12000274658203,
            ],
            self.pearsonr_parameters[0],
        )
        self.assertEqual(
            [
                1.0071699619293213,
                0.9981399774551392,
                0.9919800162315369,
                0.9980499744415283,
                0.9811400175094604,
                0.9678999781608582,
            ],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker2_default(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, data=data
        )

        # Assert
        self.assertEqual(
            [
                91.79000091552734,
                88.91000366210938,
                85.83000183105469,
                86.47000122070312,
                88.95999908447266,
                88.12000274658203,
            ],
            self.pearsonr_parameters[0],
        )
        self.assertEqual(
            [
                1.0071699619293213,
                0.9981399774551392,
                0.9919800162315369,
                0.9980499744415283,
                0.9811400175094604,
                0.9678999781608582,
            ],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker1_not_in_data_columns(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Settlement Price"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_column_ticker2_not_in_data_columns(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Settlement Price"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_ticker1_not_in_data_columns(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "GC=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception), "Parameters 'ticker1' and 'ticker2' must represent valid tickers in the 'data'."
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_ticker2_not_in_data_columns(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "CADUSD=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_single_combination(
                ticker1=ticker1,
                ticker2=ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
                data=data,
            )
        self.assertEqual(
            str(e.exception), "Parameters 'ticker1' and 'ticker2' must represent valid tickers in the 'data'."
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_data_none(self, mock_get_data_method, mock_pearsonr_method):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        data = None
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, data=data
        )

        # Assert
        get_data_parameters_expected = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.TEN_YEARS,
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(self.get_data_parameters, get_data_parameters_expected)

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_data_default(self, mock_get_data_method, mock_pearsonr_method):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1)

        # Assert
        get_data_parameters_expected = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.TEN_YEARS,
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(self.get_data_parameters, get_data_parameters_expected)

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_nan_in_data_ticker1(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        data_slice = data.loc["2022-11-08", "Close"]
        data_slice["CL=F"] = math.nan
        data.loc["2022-11-08", "Close"] = data_slice
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertEqual(
            [91.79000091552734, 85.83000183105469, 86.47000122070312, 88.95999908447266, 88.12000274658203],
            self.pearsonr_parameters[0],
        )
        self.assertEqual(
            [1.0071699619293213, 0.9919800162315369, 0.9980499744415283, 0.9811400175094604, 0.9678999781608582],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_nan_in_data_ticker2(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        data_slice = data.loc["2022-11-09", "Close"]
        data_slice["EUR=X"] = math.nan
        data.loc["2022-11-09", "Close"] = data_slice
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertEqual(
            [91.79000091552734, 88.91000366210938, 86.47000122070312, 88.95999908447266, 88.12000274658203],
            self.pearsonr_parameters[0],
        )
        self.assertEqual(
            [1.0071699619293213, 0.9981399774551392, 0.9980499744415283, 0.9811400175094604, 0.9678999781608582],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_nan_in_data_ticker1_and_data_ticker2(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Close"
        column_ticker2 = "Close"
        data = self.data
        data_slice_1 = data.loc["2022-11-08", "Close"]
        data_slice_1["CL=F"] = math.nan
        data.loc["2022-11-08", "Close"] = data_slice_1
        data_slice_2 = data.loc["2022-11-09", "Close"]
        data_slice_2["EUR=X"] = math.nan
        data.loc["2022-11-09", "Close"] = data_slice_2
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertEqual(
            [91.79000091552734, 86.47000122070312, 88.95999908447266, 88.12000274658203], self.pearsonr_parameters[0]
        )
        self.assertEqual(
            [1.0071699619293213, 0.9980499744415283, 0.9811400175094604, 0.9678999781608582],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_no_nan_in_data(self, mock_get_data_method, mock_pearsonr_method):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertFalse(hasattr(self, "get_data_parameters"))
        self.assertEqual([322419, 344223, 388301, 340007, 340007, 116673], self.pearsonr_parameters[0])
        self.assertEqual(
            [
                1.0071699619293213,
                0.9981399774551392,
                0.9919800162315369,
                0.9980499744415283,
                0.9811400175094604,
                0.9678999781608582,
            ],
            self.pearsonr_parameters[1],
        )

    @patch("scipy.stats.pearsonr")
    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_single_combination_returns_float_tuple_of_length_two(
        self, mock_get_data_method, mock_pearsonr_method
    ):

        # Arrange
        self.load_data_single_combination()
        ticker1 = "CL=F"
        ticker2 = "EUR=X"
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        data = self.data
        mock_get_data_method.side_effect = self.mock_get_data_side_effect
        mock_pearsonr_method.side_effect = self.mock_pearsonr_side_effect

        # Act
        returned_value = correlation_analysis_single_combination(
            ticker1=ticker1, ticker2=ticker2, column_ticker1=column_ticker1, column_ticker2=column_ticker2, data=data
        )

        # Assert
        self.assertIsInstance(returned_value, tuple)
        self.assertEqual(2, len(returned_value))
        pearson_corr_coef = returned_value[0]
        self.assertTrue(-1 <= pearson_corr_coef <= 1)
        p_value = returned_value[1]
        self.assertTrue(0 <= p_value <= 1)

    # Tests for method correlation_analysis_lists_cardinal_product()

    def load_data_cardinal_product(self):
        with open(f"{self.TEST_DATA_DIR}/yf_download_cardinal_product_output.pickle", "rb") as file:
            self.data = pickle.load(file)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_list_ticker1_empty_list(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = []
        list_ticker2 = ["EUR=X", "CADUSD=X", "GBP=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        correlations_expected = {}
        self.assertEqual(correlations_expected, correlations)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_list_ticker2_empty_list(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F", "GC=F"]
        list_ticker2 = []
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        correlations_expected = {}
        self.assertEqual(correlations_expected, correlations)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_single_combination(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F"]
        list_ticker2 = ["EUR=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        self.assertEqual(1, len(correlations))
        combinations_expected = {("CL=F", "EUR=X")}
        self.assertEqual(combinations_expected, set(correlations.keys()))
        for correlation in correlations.values():
            self.assertIsInstance(correlation, tuple)
            self.assertEqual(2, len(correlation))
            pearson_corr_coef, p_value = correlation
            self.assertIsInstance(pearson_corr_coef, float)
            self.assertTrue(-1 <= pearson_corr_coef <= 1)
            self.assertIsInstance(p_value, float)
            self.assertTrue(0 <= p_value <= 1)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_list_ticker1_length_one(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F"]
        list_ticker2 = ["EUR=X", "CADUSD=X", "GBP=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        self.assertEqual(3, len(correlations))
        combinations_expected = {("CL=F", "EUR=X"), ("CL=F", "CADUSD=X"), ("CL=F", "GBP=X")}
        self.assertEqual(combinations_expected, set(correlations.keys()))
        for correlation in correlations.values():
            self.assertIsInstance(correlation, tuple)
            self.assertEqual(2, len(correlation))
            pearson_corr_coef, p_value = correlation
            self.assertIsInstance(pearson_corr_coef, float)
            self.assertTrue(-1 <= pearson_corr_coef <= 1)
            self.assertIsInstance(p_value, float)
            self.assertTrue(0 <= p_value <= 1)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_list_ticker2_length_one(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F", "GC=F"]
        list_ticker2 = ["EUR=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        self.assertEqual(2, len(correlations))
        combinations_expected = {("CL=F", "EUR=X"), ("GC=F", "EUR=X")}
        self.assertEqual(combinations_expected, set(correlations.keys()))
        for correlation in correlations.values():
            self.assertIsInstance(correlation, tuple)
            self.assertEqual(2, len(correlation))
            pearson_corr_coef, p_value = correlation
            self.assertIsInstance(pearson_corr_coef, float)
            self.assertTrue(-1 <= pearson_corr_coef <= 1)
            self.assertIsInstance(p_value, float)
            self.assertTrue(0 <= p_value <= 1)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_list_ticker1_list_ticker2_length_over_one(
        self, mock_get_data_method
    ):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F", "GC=F"]
        list_ticker2 = ["EUR=X", "CADUSD=X", "GBP=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act
        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=list_ticker1,
            list_ticker2=list_ticker2,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
        )

        # Assert
        self.assertEqual(6, len(correlations))
        combinations_expected = {
            ("CL=F", "EUR=X"),
            ("CL=F", "CADUSD=X"),
            ("CL=F", "GBP=X"),
            ("GC=F", "EUR=X"),
            ("GC=F", "CADUSD=X"),
            ("GC=F", "GBP=X"),
        }
        self.assertEqual(combinations_expected, set(correlations.keys()))
        for correlation in correlations.values():
            self.assertIsInstance(correlation, tuple)
            self.assertEqual(2, len(correlation))
            pearson_corr_coef, p_value = correlation
            self.assertIsInstance(pearson_corr_coef, float)
            self.assertTrue(-1 <= pearson_corr_coef <= 1)
            self.assertIsInstance(p_value, float)
            self.assertTrue(0 <= p_value <= 1)

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_column_ticker1_not_in_data(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F", "GC=F"]
        list_ticker2 = ["EUR=X", "CADUSD=X", "GBP=X"]
        column_ticker1 = "Settlement Price"
        column_ticker2 = "Close"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_lists_cardinal_product(
                list_ticker1=list_ticker1,
                list_ticker2=list_ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_correlation_analysis_lists_cardinal_product_column_ticker2_not_in_data(self, mock_get_data_method):

        # Arrange
        self.load_data_cardinal_product()
        list_ticker1 = ["CL=F", "GC=F"]
        list_ticker2 = ["EUR=X", "CADUSD=X", "GBP=X"]
        column_ticker1 = "Volume"
        column_ticker2 = "Settlement Price"
        mock_get_data_method.side_effect = self.mock_get_data_side_effect

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            correlation_analysis_lists_cardinal_product(
                list_ticker1=list_ticker1,
                list_ticker2=list_ticker2,
                column_ticker1=column_ticker1,
                column_ticker2=column_ticker2,
            )
        self.assertEqual(
            str(e.exception),
            "Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.",
        )
