"""Class to get financial data from Yahoo Finance, using the yfinance Python
API. """

from typing import List, Union

import pandas as pd
import yfinance as yf

from src.tools.constants import (
    YfinanceGroupBy,
    YfinanceInterval,
    YfinancePeriod,
)


class YfinanceDataProvider:
    """Class yfinance data provider."""

    @staticmethod
    def get_data(
        tickers: Union[str, List[str]],
        period: Union[YfinancePeriod, str],
        interval: Union[YfinanceInterval, str],
        group_by: Union[YfinanceGroupBy, str] = YfinanceGroupBy.COLUMN,
    ) -> pd.DataFrame:
        """Get historical prices data from Yahoo Finance.

        Args:
            tickers (Union[str, List[str]]): The ticker for the asset(s) to
                retrieve historical prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.
            interval (Union[YfinanceInterval, str]): The size of the
                interval between each data point.
            group_by (Union[YfinanceGroupBy, str]): Group values in df by
                'column' or 'ticker' if getting data for multiples tickers.

        Returns:
            data (pd.DataFrame): The historical prices time series, as returned
                by yfinance.

        """

        if not tickers:
            raise ValueError("Parameter 'tickers' cannot be empty.")

        if isinstance(period, YfinancePeriod):
            period = period.value
        if isinstance(interval, YfinanceInterval):
            interval = interval.value
        if isinstance(group_by, YfinanceGroupBy):
            group_by = group_by.value

        # An invalid request to yf.download() will return a pandas DataFrame
        # with named columns but empty values (no rows).
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        return data

    @staticmethod
    def get_daily_close_prices(
        tickers: Union[str, List[str]],
        period: Union[YfinancePeriod, str] = YfinancePeriod.MAX,
    ) -> pd.DataFrame:
        """Get historical daily Close prices for tickers, from Yahoo Finance.

        Args:
            tickers (Union[str, List[str]]): The ticker for the asset(s) to
                retrieve daily historical Close prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.

        Returns:
            close_data (pd.DataFrame): The historical Close prices time series.

        """

        raw_data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=YfinanceInterval.ONE_DAY,
            group_by=YfinanceGroupBy.COLUMN,
        )
        close_data = raw_data["Close"]
        if isinstance(close_data, pd.Series):
            ticker = tickers
            if isinstance(tickers, list):
                ticker = tickers[0]
            close_data = pd.DataFrame({ticker: close_data})
        return close_data
