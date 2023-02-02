"""Class to get financial data from Yahoo Finance, using the yfinance Python API."""

from typing import List, Union

import pandas as pd
import yfinance as yf

from src.tools.constants import YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.helper_methods import extract_returns_from_dataframe


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
            tickers (Union[str, List[str]]): The ticker for the asset(s) to retrieve historical prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.
            interval (Union[YfinanceInterval, str]): The size of the interval between each data point.
            group_by (Union[YfinanceGroupBy, str]): Group values in df by 'column' or 'ticker' if getting data for
                multiple tickers.

        Returns:
            data (pd.DataFrame): The historical prices time series, as returned by yfinance.

        """

        if not tickers:
            raise ValueError("Parameter 'tickers' cannot be empty.")

        if isinstance(period, YfinancePeriod):
            period = period.value
        if isinstance(interval, YfinanceInterval):
            interval = interval.value
        if isinstance(group_by, YfinanceGroupBy):
            group_by = group_by.value

        # Invalid request to yf.download() will return a pandas DataFrame with named columns but empty values (no rows).
        data = yf.download(
            tickers=tickers, period=period, interval=interval, group_by=group_by, ignore_tz=False, progress=False
        )

        return data

    @staticmethod
    def get_hourly_returns(
        tickers: Union[str, List[str]],
        period: Union[YfinancePeriod, str] = YfinancePeriod.SEVEN_HUNDRED_TWENTY_NINE_DAYS,
    ) -> pd.DataFrame:
        """Get historical hourly Open and Close prices for tickers, from Yahoo Finance, and calculate hourly returns.

        Args:
            tickers (Union[str, List[str]]): The ticker for the asset(s) to retrieve hourly historical Open and Close
                prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.

        Returns:
            returns_data (pd.DataFrame): The calculated hourly returns time series.

        """

        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=YfinanceInterval.ONE_HOUR,
            group_by=YfinanceGroupBy.TICKER,
        )

        if isinstance(tickers, list) and len(tickers) > 1:
            returns_data = pd.DataFrame()
            for ticker in tickers:
                ticker_returns_data = extract_returns_from_dataframe(data=data[ticker])
                returns_data = pd.concat(
                    [
                        returns_data,
                        pd.DataFrame(data={ticker: ticker_returns_data}),
                    ],
                    ignore_index=False,
                    axis=1,
                )
        else:
            if isinstance(tickers, list):
                tickers = tickers[0]
            return pd.DataFrame(data={tickers: extract_returns_from_dataframe(data=data)})
        return returns_data
