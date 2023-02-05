"""Script to test the correlation of all combinations (cardinal product) between a list of commodity tickers and a list
of foreign exchange tickers. Aims to find relevant relationships between arbitrary combinations of commodity and foreign
exchange tickers."""

from src.tools.constants import PriceAttribute
from src.tools.correlation_analysis import correlation_analysis_lists_cardinal_product

if __name__ == "__main__":

    """
    commodity_list: List of arbitrary commodity tickers, to analyse their correlation with foreign exchange tickers.
    forex_list: List of arbitrary foreign exchange tickers, to analyse their correlation with commodity tickers.
    """

    commodity_list = [
        "CL=F",
        "BZ=F",
        "PL=F",
        "GC=F",
        "SI=F",
        "HG=F",
        "PA=F",
        "NG=F",
        "HO=F",
        "RB=F",
        "ZC=F",
        "ZO=F",
        "KE=F",
        "ZR=F",
        "ZM=F",
        "ZL=F",
        "ZS=F",
        "GF=F",
        "HE=F",
        "LE=F",
        "CC=F",
        "KC=F",
        "CT=F",
        "LBS=F",
        "OJ=F",
        "SB=F",
    ]
    forex_list = [
        "AUDCAD=X",
        "AUDCHF=X",
        "AUDJPY=X",
        "AUDUSD=X",
        "CADCHF=X",
        "CADJPY=X",
        "CHFJPY=X",
        "EURAUD=X",
        "EURCAD=X",
        "EURCHF=X",
        "EURGBP=X",
        "EURJPY=X",
        "EURUSD=X",
        "GBPAUD=X",
        "GBPCAD=X",
        "GBPCHF=X",
        "GBPJPY=X",
        "GBPUSD=X",
        "USDCAD=X",
        "USDCHF=X",
        "USDJPY=X",
        "EURHUF=X",
        "USDCNY=X",
        "USDHKD=X",
        "USDSGD=X",
        "USDINR=X",
        "USDMXN=X",
        "USDPHP=X",
        "USDIDR=X",
        "USDTHB=X",
        "USDMYR=X",
        "USDZAR=X",
        "USDRUB=X",
    ]
    columns = [
        (PriceAttribute.CLOSE, PriceAttribute.CLOSE, 0.85),
        (PriceAttribute.HIGH, PriceAttribute.CLOSE, 0.85),
        (PriceAttribute.LOW, PriceAttribute.CLOSE, 0.85),
        (PriceAttribute.HIGH, PriceAttribute.HIGH, 0.85),
        (PriceAttribute.LOW, PriceAttribute.HIGH, 0.85),
        (PriceAttribute.CLOSE, PriceAttribute.HIGH, 0.85),
        (PriceAttribute.LOW, PriceAttribute.LOW, 0.85),
        (PriceAttribute.HIGH, PriceAttribute.LOW, 0.85),
        (PriceAttribute.CLOSE, PriceAttribute.LOW, 0.85),
        (PriceAttribute.VOLUME, PriceAttribute.CLOSE, 0.13),
    ]

    for column_commodity, column_forex, correlation_threshold in columns:
        print(f"\n{column_commodity.value} -> {column_forex.value}\n")

        correlations = correlation_analysis_lists_cardinal_product(
            list_ticker1=commodity_list,
            list_ticker2=forex_list,
            column_ticker1=column_commodity,
            column_ticker2=column_forex,
        )

        # Print correlation coefficients sorted from most correlated to least, if correlation > correlation_threshold
        for combination, correlation in {
            combination: correlation
            for combination, correlation in sorted(correlations.items(), key=lambda item: -abs(item[1][0]))
            if abs(correlation[0]) > correlation_threshold
        }.items():
            print(
                f"{combination[0]} -> {combination[1]} : ({correlation[0]}, {correlation[1]}), "
                f"data length: {correlation[2]}"
            )
