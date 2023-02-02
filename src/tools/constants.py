"""Constant Enum classes."""

from enum import Enum


class YfinancePeriod(Enum):
    """Period for yfinance.download()."""

    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    SIXTY_DAYS = "60d"
    SEVEN_HUNDRED_TWENTY_NINE_DAYS = "729d"
    SEVEN_HUNDRED_THIRTY_DAYS = "730d"
    ONE_MONTH = "1mo"
    ONE_YEAR = "1y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    TWENTY_YEARS = "20y"
    MAX = "max"


class YfinanceInterval(Enum):
    """Interval for yfinance.download()."""

    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    NINETY_MINUTES = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class YfinanceGroupBy(Enum):
    """Formatting options for yfinance.download() with multiple assets."""

    COLUMN = "column"
    TICKER = "ticker"
