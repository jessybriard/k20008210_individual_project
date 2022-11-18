"""Method for statistical analysis of the correlation between two tickers."""

import math
from typing import Tuple, Union

import pandas as pd
from scipy.stats.stats import pearsonr

from src.tools.constants import YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.yfinance_data_provider import YfinanceDataProvider


def correlation_analysis(
    ticker1: str,
    ticker2: str,
    column_ticker1: str = "Close",
    column_ticker2: str = "Close",
    data: Union[None, pd.DataFrame] = None,
) -> Tuple[float, float]:
    """Correlation analysis of historical data for two tickers. Returns the Pearson correlation coefficient
    (Pearson's r) and p-value between the two datasets. A Pearson correlation coefficient ranges between -1 and 1, with
    0 implying no correlation, 1 implying an exact positive linear correlation and -1 implying an exact negative linear
    correlation.

    Args:
        ticker1 (str): The first ticker to compare.
        ticker2 (str): The second ticker to compare.
        column_ticker1 (str): The attribute to use as data for ticker1 (the column of the 'data' DataFrame).
        column_ticker2 (str): The attribute to use as data for ticker2 (the column of the 'data' DataFrame).
        data (Union[None, pd.DataFrame]): The DataFrame containing the historical data for the tickers ; if not
            provided, download 10 years of past daily historical data for the tickers.

    Returns:
        pearson_corr_coef (float): The Pearson correlation coefficient (or "Pearson's r") between the two datasets.
        p_value (float): The p-value between the two datasets, for this Pearson correlation coefficient.

    """
    if not ticker1 or not ticker2 or ticker1 == ticker2:
        raise ValueError("Parameters 'ticker1' and 'ticker2' must be non-empty.")
    if ticker1 == ticker2:
        raise ValueError("Parameters 'ticker1' and 'ticker2' must represent different assets.")
    if data is None:
        data = YfinanceDataProvider.get_data(
            tickers=[ticker1, ticker2],
            period=YfinancePeriod.TEN_YEARS,
            interval=YfinanceInterval.ONE_DAY,
            group_by=YfinanceGroupBy.COLUMN,
        )
    if column_ticker1 not in data.columns or column_ticker2 not in data.columns:
        raise ValueError("Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.")
    data_ticker1 = data[column_ticker1][ticker1].values
    data_ticker2 = data[column_ticker2][ticker2].values
    non_nan_rows = [i for i in range(len(data)) if not math.isnan(data_ticker1[i]) and not math.isnan(data_ticker2[i])]
    clean_data_ticker1 = [data_ticker1[i] for i in non_nan_rows]
    clean_data_ticker2 = [data_ticker2[i] for i in non_nan_rows]
    return pearsonr(clean_data_ticker1, clean_data_ticker2)
