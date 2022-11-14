"""Class to get financial data from Yahoo Finance, using the yfinance Python
API. """

from typing import List, Union

import pandas as pd
import yfinance as yf

from main.tools.constants import YfinanceCandleSize, YfinancePeriod


class YfinanceDataProvider:
    """Class yfinance data provider."""

    @staticmethod
    def get_data(
        assets: Union[str, List[str]],
        period: Union[YfinancePeriod, str],
        candle_size: YfinanceCandleSize,
        group_by: str = "column",
    ) -> pd.DataFrame:
        """Get historical prices data from yfinance.

        Args:
            assets (Union[str, List[str]]): The code for the asset(s) to
                retrieve historical prices for.
            period (Union[YfinancePeriod, str]): The period of the time series.
            candle_size (YfinanceCandleSize): The size of the interval between
                each data point.
            group_by (str): Group values in df by 'column' or 'ticker' if
                getting data for multiples tickers.

        Returns:
            data (pd.DataFrame): The historical prices time series, as returned
                by yfinance.

        """
        if isinstance(period, YfinancePeriod):
            period = period.value
        if isinstance(candle_size, YfinanceCandleSize):
            candle_size = candle_size.value
        data = yf.download(
            tickers=assets,
            period=period,
            interval=candle_size,
            group_by=group_by,
        )
        return data
