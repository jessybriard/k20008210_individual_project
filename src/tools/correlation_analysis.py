"""Method for statistical analysis of the correlation between two tickers."""

import math
from itertools import product
from typing import List, Tuple, Union

import pandas as pd
from scipy.stats.stats import pearsonr

from src.tools.constants import YfinanceGroupBy, YfinanceInterval, YfinancePeriod
from src.tools.yfinance_data_provider import YfinanceDataProvider


def correlation_analysis_single_combination(
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
    attribute_columns = {attribute for attribute, ticker in data.columns}
    if column_ticker1 not in attribute_columns or column_ticker2 not in attribute_columns:
        raise ValueError("Parameters 'column_ticker1' and 'column_ticker2' must represent valid columns of the 'data'.")
    if ticker1 not in data[column_ticker1].columns or ticker2 not in data[column_ticker2].columns:
        raise ValueError("Parameters 'ticker1' and 'ticker2' must represent valid tickers in the 'data'.")
    data_ticker1 = data[column_ticker1][ticker1].values
    data_ticker2 = data[column_ticker2][ticker2].values
    non_nan_rows = [i for i in range(len(data)) if not math.isnan(data_ticker1[i]) and not math.isnan(data_ticker2[i])]
    clean_data_ticker1 = [data_ticker1[i] for i in non_nan_rows]
    clean_data_ticker2 = [data_ticker2[i] for i in non_nan_rows]
    return tuple(pearsonr(clean_data_ticker1, clean_data_ticker2))


def correlation_analysis_lists_cardinal_product(
    list_ticker1: List[str], list_ticker2: List[str], column_ticker1: str, column_ticker2: str
) -> dict:
    """Analyse the correlation of all combinations (cardinal product) between two lists of tickers, for specific
    attributes. Returns a dictionary containing correlation insights for all ticker combinations.

    Args:
        list_ticker1 (List[str]): First list of tickers, to analyse their correlation with the second list of tickers.
        list_ticker2 (List[str]): Second list of tickers, to analyse their correlation with the first list of tickers.
        column_ticker1 (str): The attribute of the ticker from the first list to use to analyse correlation.
        column_ticker2 (str): The attribute of the ticker from the second list to use to analyse correlation.

    Returns:
        correlation_insights (dict): Dictionary containing correlation insights for all ticker combinations.
    """
    data = YfinanceDataProvider.get_data(
        tickers=list_ticker1 + list_ticker2,
        period=YfinancePeriod.TEN_YEARS,
        interval=YfinanceInterval.ONE_DAY,
        group_by=YfinanceGroupBy.COLUMN,
    )
    combinations = list(product(list_ticker1, list_ticker2))
    correlations = {}
    for commodity_ticker, forex_ticker in combinations:
        correlations[(commodity_ticker, forex_ticker)] = correlation_analysis_single_combination(
            ticker1=commodity_ticker,
            ticker2=forex_ticker,
            column_ticker1=column_ticker1,
            column_ticker2=column_ticker2,
            data=data,
        )
    return correlations
