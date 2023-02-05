"""Class to get financial data from Yahoo Finance, using the yfinance Python API."""

from typing import List, Union

import pandas as pd
import yfinance as yf

from src.tools.constants import PriceAttribute, YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.helper_methods import extract_changes_from_dataframe


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
    def get_hourly_changes(
        attributes: List[PriceAttribute],
        tickers: List[str],
        period: Union[YfinancePeriod, str] = YfinancePeriod.SEVEN_HUNDRED_TWENTY_NINE_DAYS,
    ) -> pd.DataFrame:
        """Get historical hourly prices for tickers, from Yahoo Finance, and calculate hourly changes for the selected
        price attribute.

        Args:
            attributes (List[PriceAttribute]): The price attribute(s) (column(s)) to retrieve hourly data for.
            tickers (List[str]): The ticker for the asset(s) to retrieve hourly historical prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.

        Returns:
            changes_data (pd.DataFrame): The calculated hourly changes (percentage) time series.

        """

        if not attributes:
            raise ValueError("Parameter 'attributes' cannot be empty.")

        data = YfinanceDataProvider.get_data(
            tickers=tickers,
            period=period,
            interval=YfinanceInterval.ONE_HOUR,
            group_by=YfinanceGroupBy.TICKER,
        )

        changes_data = pd.DataFrame()
        for ticker in tickers:
            ticker_data = data[ticker] if len(tickers) > 1 else data
            for attribute in attributes:
                ticker_changes_data = extract_changes_from_dataframe(attribute=attribute, data=ticker_data)
                changes_data = pd.concat(
                    [
                        changes_data,
                        pd.DataFrame(data={(ticker, attribute.value): ticker_changes_data}),
                    ],
                    ignore_index=False,
                    axis=1,
                )
        return changes_data
