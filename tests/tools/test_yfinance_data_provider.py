"""Tests for methods in file yfinance_data_provider.py."""

import os
import pickle
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src.tools.constants import YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.yfinance_data_provider import YfinanceDataProvider


class TestYfinanceDataProvider(TestCase):
    """Test class for methods in class YfinanceDataProvider."""

    def setUp(self) -> None:
        self.TEST_DATA_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/test_data"

    def mock_download_side_effect(self, **kwargs):
        self.parameters = kwargs
        tickers = kwargs["tickers"]
        if isinstance(tickers, list) and len(tickers) > 1:  # Multiple tickers request
            if "interval" in kwargs.keys() and kwargs["interval"] == YfinanceInterval.ONE_HOUR:
                output_pickled_file_name = "yf_download_hourly_multiple_tickers_output.pickle"
            elif "group_by" in kwargs.keys() and kwargs["group_by"] == YfinanceGroupBy.TICKER:
                output_pickled_file_name = "yf_download_group_by_ticker_output.pickle"
            else:
                output_pickled_file_name = "yf_download_multiple_tickers_output.pickle"
        else:  # Single ticker request
            if "interval" in kwargs.keys() and kwargs["interval"] == YfinanceInterval.ONE_HOUR:
                output_pickled_file_name = "yf_download_hourly_single_ticker_output.pickle"
            else:
                output_pickled_file_name = "yf_download_single_ticker_output.pickle"
        with open(
            f"{self.TEST_DATA_DIR}/{output_pickled_file_name}",
            "rb",
        ) as file:
            self.yf_download_output = pickle.load(file)
        return self.yf_download_output

    # Tests for method get_data()

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
            "ignore_tz": False,
            "progress": False,
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
            "ignore_tz": False,
            "progress": False,
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
            "ignore_tz": False,
            "progress": False,
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
        self.assertEqual(str(e.exception), "Parameter 'tickers' cannot be empty.")

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
        self.assertEqual(str(e.exception), "Parameter 'tickers' cannot be empty.")

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
            "ignore_tz": False,
            "progress": False,
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
            "ignore_tz": False,
            "progress": False,
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
            "ignore_tz": False,
            "progress": False,
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
        data = YfinanceDataProvider.get_data(tickers=tickers, period=period, interval=interval)

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "1wk",
            "interval": "1d",
            "group_by": "column",
            "ignore_tz": False,
            "progress": False,
        }
        self.assertEqual(expected_parameters, self.parameters)
        self.assertTrue(self.yf_download_output.equals(data))

    # Tests for method get_hourly_changes()

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_hourly_changes_single_ticker_str(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = "CL=F"
        period = "7h"

        # Act
        changes_data = YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)

        # Assert
        expected_parameters = {
            "tickers": "CL=F",
            "period": "7h",
            "interval": YfinanceInterval.ONE_HOUR,
            "group_by": YfinanceGroupBy.TICKER,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_changes_series = pd.Series(
            data={
                "2023-02-01 02:00:00-05:00": 0.0006307774541483123,
                "2023-02-01 03:00:00-05:00": 0.003026454659943564,
                "2023-02-01 04:00:00-05:00": -0.007793874579472201,
                "2023-02-01 05:00:00-05:00": -0.00025341109503747235,
                "2023-02-01 06:00:00-05:00": 0.006208005344925863,
                "2023-02-01 07:00:00-05:00": -0.002896402799731141,
                "2023-02-01 08:00:00-05:00": 0.0013892474100479647,
            }
        )
        expected_changes_series.index = pd.DatetimeIndex(
            expected_changes_series.index, dtype="datetime64[ns, America/New_York]"
        )
        expected_changes_data = pd.DataFrame(data={"CL=F": expected_changes_series})
        self.assertTrue(expected_changes_data.equals(changes_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_hourly_changes_single_ticker_list(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F"]
        period = "7h"

        # Act
        changes_data = YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)

        # Assert
        expected_parameters = {
            "tickers": ["CL=F"],
            "period": "7h",
            "interval": YfinanceInterval.ONE_HOUR,
            "group_by": YfinanceGroupBy.TICKER,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_changes_series = pd.Series(
            data={
                "2023-02-01 02:00:00-05:00": 0.0006307774541483123,
                "2023-02-01 03:00:00-05:00": 0.003026454659943564,
                "2023-02-01 04:00:00-05:00": -0.007793874579472201,
                "2023-02-01 05:00:00-05:00": -0.00025341109503747235,
                "2023-02-01 06:00:00-05:00": 0.006208005344925863,
                "2023-02-01 07:00:00-05:00": -0.002896402799731141,
                "2023-02-01 08:00:00-05:00": 0.0013892474100479647,
            }
        )
        expected_changes_series.index = pd.DatetimeIndex(
            expected_changes_series.index, dtype="datetime64[ns, America/New_York]"
        )
        expected_changes_data = pd.DataFrame(data={"CL=F": expected_changes_series})
        self.assertTrue(expected_changes_data.equals(changes_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_hourly_changes_multiple_tickers_list(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = "7h"

        # Act
        changes_data = YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": "7h",
            "interval": YfinanceInterval.ONE_HOUR,
            "group_by": YfinanceGroupBy.TICKER,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_changes_series_cl = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": 0.0030352581850995137,
                "2023-02-01 08:00:00+00:00": 0.003026454659943564,
                "2023-02-01 09:00:00+00:00": -0.007793874579472201,
                "2023-02-01 10:00:00+00:00": -0.00025341109503747235,
                "2023-02-01 11:00:00+00:00": 0.006208005344925863,
                "2023-02-01 12:00:00+00:00": -0.002896402799731141,
                "2023-02-01 13:00:00+00:00": 0.0010103792718659285,
            }
        )
        expected_changes_series_cl.index = pd.DatetimeIndex(expected_changes_series_cl.index)
        expected_changes_series_eur = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": -0.0010873618584464582,
                "2023-02-01 08:00:00+00:00": -0.0005442078713167677,
                "2023-02-01 09:00:00+00:00": 0.00043564230462633823,
                "2023-02-01 10:00:00+00:00": -0.0008709052061015792,
                "2023-02-01 11:00:00+00:00": -0.0003267866914642771,
                "2023-02-01 12:00:00+00:00": 0.00021797231063184626,
                "2023-02-01 13:00:00+00:00": -0.001307354046709806,
            }
        )
        expected_changes_series_eur.index = pd.DatetimeIndex(expected_changes_series_eur.index)
        expected_changes_data = pd.DataFrame(
            data={
                "CL=F": expected_changes_series_cl,
                "EUR=X": expected_changes_series_eur,
            }
        )
        self.assertTrue(expected_changes_data.equals(changes_data))

    @patch("yfinance.download")
    def test_get_hourly_changes_tickers_empty_str(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = ""
        period = "7h"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)
        self.assertEqual(str(e.exception), "Parameter 'tickers' cannot be empty.")

    @patch("yfinance.download")
    def test_get_hourly_changes_tickers_empty_list(self, mock_download_method):

        # Arrange
        mock_download_method.side_effect = self.mock_download_side_effect
        tickers = []
        period = "7h"

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)
        self.assertEqual(str(e.exception), "Parameter 'tickers' cannot be empty.")

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_hourly_changes_period_enum(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]
        period = YfinancePeriod.ONE_DAY

        # Act
        changes_data = YfinanceDataProvider.get_hourly_changes(tickers=tickers, period=period)

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.ONE_DAY,
            "interval": YfinanceInterval.ONE_HOUR,
            "group_by": YfinanceGroupBy.TICKER,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_changes_series_cl = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": 0.0030352581850995137,
                "2023-02-01 08:00:00+00:00": 0.003026454659943564,
                "2023-02-01 09:00:00+00:00": -0.007793874579472201,
                "2023-02-01 10:00:00+00:00": -0.00025341109503747235,
                "2023-02-01 11:00:00+00:00": 0.006208005344925863,
                "2023-02-01 12:00:00+00:00": -0.002896402799731141,
                "2023-02-01 13:00:00+00:00": 0.0010103792718659285,
            }
        )
        expected_changes_series_cl.index = pd.DatetimeIndex(expected_changes_series_cl.index)
        expected_changes_series_eur = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": -0.0010873618584464582,
                "2023-02-01 08:00:00+00:00": -0.0005442078713167677,
                "2023-02-01 09:00:00+00:00": 0.00043564230462633823,
                "2023-02-01 10:00:00+00:00": -0.0008709052061015792,
                "2023-02-01 11:00:00+00:00": -0.0003267866914642771,
                "2023-02-01 12:00:00+00:00": 0.00021797231063184626,
                "2023-02-01 13:00:00+00:00": -0.001307354046709806,
            }
        )
        expected_changes_series_eur.index = pd.DatetimeIndex(expected_changes_series_eur.index)
        expected_changes_data = pd.DataFrame(
            data={
                "CL=F": expected_changes_series_cl,
                "EUR=X": expected_changes_series_eur,
            }
        )
        self.assertTrue(expected_changes_data.equals(changes_data))

    @patch("src.tools.yfinance_data_provider.YfinanceDataProvider.get_data")
    def test_get_hourly_changes_period_default(self, mock_get_data_method):

        # Arrange
        mock_get_data_method.side_effect = self.mock_download_side_effect
        tickers = ["CL=F", "EUR=X"]

        # Act
        changes_data = YfinanceDataProvider.get_hourly_changes(tickers=tickers)

        # Assert
        expected_parameters = {
            "tickers": ["CL=F", "EUR=X"],
            "period": YfinancePeriod.SEVEN_HUNDRED_TWENTY_NINE_DAYS,
            "interval": YfinanceInterval.ONE_HOUR,
            "group_by": YfinanceGroupBy.TICKER,
        }
        self.assertEqual(expected_parameters, self.parameters)
        expected_changes_series_cl = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": 0.0030352581850995137,
                "2023-02-01 08:00:00+00:00": 0.003026454659943564,
                "2023-02-01 09:00:00+00:00": -0.007793874579472201,
                "2023-02-01 10:00:00+00:00": -0.00025341109503747235,
                "2023-02-01 11:00:00+00:00": 0.006208005344925863,
                "2023-02-01 12:00:00+00:00": -0.002896402799731141,
                "2023-02-01 13:00:00+00:00": 0.0010103792718659285,
            }
        )
        expected_changes_series_cl.index = pd.DatetimeIndex(expected_changes_series_cl.index)
        expected_changes_series_eur = pd.Series(
            data={
                "2023-02-01 07:00:00+00:00": -0.0010873618584464582,
                "2023-02-01 08:00:00+00:00": -0.0005442078713167677,
                "2023-02-01 09:00:00+00:00": 0.00043564230462633823,
                "2023-02-01 10:00:00+00:00": -0.0008709052061015792,
                "2023-02-01 11:00:00+00:00": -0.0003267866914642771,
                "2023-02-01 12:00:00+00:00": 0.00021797231063184626,
                "2023-02-01 13:00:00+00:00": -0.001307354046709806,
            }
        )
        expected_changes_series_eur.index = pd.DatetimeIndex(expected_changes_series_eur.index)
        expected_changes_data = pd.DataFrame(
            data={
                "CL=F": expected_changes_series_cl,
                "EUR=X": expected_changes_series_eur,
            }
        )
        self.assertTrue(expected_changes_data.equals(changes_data))
