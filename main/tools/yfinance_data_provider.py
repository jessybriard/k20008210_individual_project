"""Class to get financial data from Yahoo Finance, using the yfinance Python
API. """

from typing import List, Union

import pandas as pd
import yfinance as yf

from main.tools.constants import (
    YfinanceGroupBy,
    YfinanceInterval,
    YfinancePeriod,
)


class YfinanceDataProvider:
    """Class yfinance data provider."""

    @staticmethod
    def get_data(
        assets: Union[str, List[str]],
        period: Union[YfinancePeriod, str],
        interval: Union[YfinanceInterval, str],
        group_by: Union[YfinanceGroupBy, str] = YfinanceGroupBy.COLUMN,
    ) -> pd.DataFrame:
        """Get historical prices data from yfinance.

        Args:
            assets (Union[str, List[str]]): The code for the asset(s) to
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

        if isinstance(period, YfinancePeriod):
            period = period.value
        if isinstance(interval, YfinanceInterval):
            interval = interval.value
        if isinstance(group_by, YfinanceGroupBy):
            group_by = group_by.value

        # An invalid request to yf.download() will return a pandas DataFrame
        # with named columns but empty values (no rows).
        data = yf.download(
            tickers=assets,
            period=period,
            interval=interval,
            group_by=group_by,
        )

        return data
