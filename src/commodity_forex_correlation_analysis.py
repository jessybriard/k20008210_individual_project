"""Script to test the correlation of all combinations (cardinal product) between a list of commodity tickers and a list
of foreign exchange tickers. Aims to find relevant relationships between arbitrary combinations of commodity and foreign
exchange tickers."""

from src.tools.correlation_analysis import correlation_analysis_lists_cardinal_product

if __name__ == "__main__":

    """
    commodity_list: List of arbitrary commodity tickers, to analyse their correlation with foreign exchange tickers.
    forex_list: List of arbitrary foreign exchange tickers, to analyse their correlation with commodity tickers.
    """

    commodity_list = [
        "CL=F",
        "BZ=F",
        "GC=F",
        "NG=F",
        "SI=F",
        "ZW=F",
        "KE=F",
        "ZS=F",
        "ZL=F",
        "ZM=F",
        "HE=F",
        "PL=F",
        "HG=F",
        "ZC=F",
        "ALI=F",
        "SB=F",
        "CT=F",
        "KC=F",
        "CC=F",
        "PA=F",
        "RB=F",
        "GF=F",
    ]
    forex_list = [
        "EUR=X",
        "GBP=X",
        "CADUSD=X",
        "AUD=X",
        "CHF=X",
        "JPY=X",
        "NZD=X",
        "CADJPY=X",
        "EURGBP=X",
        "EURJPY=X",
        "EURCHF=X",
        "HKD=X",
        "RUB=X",
        "EURCAD=X",
        "GBPJPY=X",
        "EURAUD=X",
        "EURNZD=X",
    ]

    # List of tuples: (column_commodity, column_forex, correlation_threshold)
    columns = [
        ("Close", "Close", 0.6),
        ("Open", "Open", 0.6),
        ("High", "High", 0.6),
        ("Low", "Low", 0.6),
        ("Volume", "Close", 0.5),
    ]

    for column_commodity, column_forex, correlation_threshold in columns:
        print(f"\n{column_commodity} -> {column_forex}\n")

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
            print(f"{combination[0]} -> {combination[1]} : ({correlation[0]}, {correlation[1]})")
