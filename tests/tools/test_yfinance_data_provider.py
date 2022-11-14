"""Tests for methods in file yfinance_data_provider.py."""

import os
import pickle
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src.tools.constants import (
    YfinanceGroupBy,
    YfinanceInterval,
    YfinancePeriod,
)
from src.tools.yfinance_data_provider import YfinanceDataProvider


class TestYfinanceDataProvider(TestCase):
    """Test class for methods in class YfinanceDataProvider."""

    def setUp(self) -> None:
        self.TEST_DATA_DIR = (
            f"{os.path.dirname(os.path.abspath(__file__))}/test_data"
        )

    def mock_download_side_effect(self, **kwargs):
        self.parameters = kwargs
        tickers = kwargs["tickers"]
        # Multiple tickers request
        if isinstance(tickers, list) and len(tickers) > 1:
            with open(
                f"{self.TEST_DATA_DIR}"
                f"/yf_download_multiple_tickers_output.pickle",
                "rb",
            ) as file:
                self.yf_download_output = pickle.load(file)
        # Single ticker request
        else:
            with open(
                f"{self.TEST_DATA_DIR}"
                f"/yf_download_single_ticker_output.pickle",
                "rb",
            ) as file:
                self.yf_download_output = pickle.load(file)
        return self.yf_download_output

    @patch("yfinance.download")
    def test_get_data_single_ticker_str(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = "CL=F"
        period = "1wk"
        interval = "1d"
        group_by = "column"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": "CL=F",
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_single_ticker_list(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F"]
        period = "1wk"
        interval = "1d"
        group_by = "column"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_multiple_tickers_list(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "1wk"
        interval = "1d"
        group_by = "column"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_tickers_empty_list(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = []
        period = "1wk"
        interval = "1d"
        group_by = "column"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_data(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by=group_by,
            )
        self.assertEqual(
            str(e.exception), "Parameter 'tickers' cannot be empty."
        )

    @patch("yfinance.download")
    def test_get_data_tickers_empty_str(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ""
        period = "1wk"
        interval = "1d"
        group_by = "column"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_data(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by=group_by,
            )
        self.assertEqual(
            str(e.exception), "Parameter 'tickers' cannot be empty."
        )

    @patch("yfinance.download")
    def test_get_data_period_enum(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = YfinancePeriod.ONE_WEEK
        interval = "1d"
        group_by = "column"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertIsInstance(self.parameters["period"], str)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_interval_enum(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "1wk"
        interval = YfinanceInterval.ONE_DAY
        group_by = "column"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertIsInstance(self.parameters["interval"], str)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_group_by_enum(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "1wk"
        interval = "1d"
        group_by = YfinanceGroupBy.COLUMN

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertIsInstance(self.parameters["group_by"], str)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("yfinance.download")
    def test_get_data_group_by_default(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "1wk"
        interval = "1d"

        # Act
        data = YfinanceDataProvider.get_data(
            tickers=tickers, period=period, interval=interval
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertTrue(self.yf_download_output.equals(data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_daily_close_prices_single_ticker_str(
        self, mock_get_data_method
    ):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = "CL=F"
        period = "1wk"

        # Act
        close_data = YfinanceDataProvider.get_daily_close_prices(
            tickers=tickers, period=period
        )

        # Assert
        expected_parameters = {
            "tickers": "CL=F",
            "period": "1wk",
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_close_data = pd.DataFrame(
            {"CL=F": self.yf_download_output["Close"]}
        )
        self.assertTrue(expected_close_data.equals(close_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_daily_close_prices_single_ticker_list(
        self, mock_get_data_method
    ):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F"]
        period = "1wk"

        # Act
        close_data = YfinanceDataProvider.get_daily_close_prices(
            tickers=tickers, period=period
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F"],
            "period": "1wk",
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_close_data = pd.DataFrame(
            {"CL=F": self.yf_download_output["Close"]}
        )
        self.assertTrue(expected_close_data.equals(close_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_daily_close_prices_multiple_tickers_list(
        self, mock_get_data_method
    ):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "1wk"

        # Act
        close_data = YfinanceDataProvider.get_daily_close_prices(
            tickers=tickers, period=period
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_close_data = self.yf_download_output["Close"]
        self.assertTrue(expected_close_data.equals(close_data))

    @patch("yfinance.download")
    def test_get_daily_close_prices_tickers_empty_list(
        self, mock_download_method
    ):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = []
        period = "1wk"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_daily_close_prices(
                tickers=tickers, period=period
            )
        self.assertEqual(
            str(e.exception), "Parameter 'tickers' cannot be empty."
        )

    @patch("yfinance.download")
    def test_get_daily_close_prices_tickers_empty_str(
        self, mock_download_method
    ):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ""
        period = "1wk"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_daily_close_prices(
                tickers=tickers, period=period
            )
        self.assertEqual(
            str(e.exception), "Parameter 'tickers' cannot be empty."
        )

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_daily_close_prices_period_enum(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = YfinancePeriod.ONE_WEEK

        # Act
        close_data = YfinanceDataProvider.get_daily_close_prices(
            tickers=tickers, period=period
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.ONE_WEEK,
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_close_data = self.yf_download_output["Close"]
        self.assertTrue(expected_close_data.equals(close_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_daily_close_prices_period_default(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]

        # Act
        close_data = YfinanceDataProvider.get_daily_close_prices(
            tickers=tickers
        )

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.MAX,
            "interval": YfinanceInterval.ONE_DAY,
            "group_by": YfinanceGroupBy.COLUMN,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_close_data = self.yf_download_output["Close"]
        self.assertTrue(expected_close_data.equals(close_data))
