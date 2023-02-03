"""Method to compare the performance of individual and sector approach for a pair of forex ticker and commodities
ticker(s), and a choice of Classification model. The method uses Monte-Carlo Cross-Validation to estimate the accuracy
of predicting (discrete) hourly returns for both approaches. Then, for both the 'individual' and the 'sector' approach,
using the results from these samples, hypothesis testing is conducted to determine if the results come from Gaussian
(normal) distribution, and to determine if the difference in mean accuracy is statistically significant compared to
random guessing and compared to the other approach. We assume the accuracy of random guessing to come from a Gaussian
distribution with mean 0.5, because our labeled data is balanced with 50% of 'True' labels and 50% of 'False' labels."""

from functools import reduce
from operator import add
from typing import List

from src.tools.hypothesis_testing import lilliefors_test, one_sample_t_test, two_sample_t_test
from src.tools.labeled_data_builder.monte_carlo_cross_validation import generate_train_test_sample
from src.tools.labeled_data_builder.time_series_forecasting import create_labeled_data
from src.tools.statistical_evaluation import ClassificationEvaluation
from src.tools.yfinance_data_provider import YfinanceDataProvider


def evaluate_and_compare(forex_ticker: str, comdty_tickers: List[str], model, nb_samples: int = 100) -> None:
    """Compare the performance of individual and sector approach for a pair of forex ticker and commodities
    ticker(s), and a choice of Classification model. The method uses Monte-Carlo Cross-Validation to estimate the
    accuracy of predicting (discrete) hourly returns for both approaches. Then, for both the 'individual' and the
    'sector' approach, using the results from these samples, hypothesis testing is conducted. First, the Lilliefors
    test is conducted to determine if the results come from a Gaussian (normal) distribution, at 95% confidence. Then,
    one-sample T-test is conducted to determine if the results are statistically significant compared to random
    guessing, at 95% confidence, assuming that the accuracy of random guessing comes from a Gaussian distribution with
    mean 0.5, because our labeled data is balanced with 50% of 'True' labels and 50% of 'False' labels. Finally,
    two-sample T-test is conducted to determine if the difference in accuracy between the 'individual' and 'sector'
    approaches is statistically significant, at 95% confidence.

    Args:
        forex_ticker (str): The ticker for the foreign exchange asset we want to predict for.
        comdty_tickers (List[str]): The ticker(s) for the commodities asset(s) we want to use as features to predict the
            foreign exchange asset.
        model: Instance of a scikit-learn Classification model, supporting methods fit() and predict().
        nb_samples (int): The number of samples to generate from the labeled data, using Monte-Carlo Cross-Validation.

    """

    features_length = 5
    data = YfinanceDataProvider.get_hourly_returns(comdty_tickers + [forex_ticker])
    labeled_data = create_labeled_data(
        ticker_label=forex_ticker,
        tickers_features=comdty_tickers + [forex_ticker],
        data=data,
        features_length=features_length,
    )

    accuracies = {"individual": [], "sector": []}
    for i in range(nb_samples):
        train_data, test_data = generate_train_test_sample(data=labeled_data, train_percentage=0.8)
        for approach in ["individual", "sector"]:
            model.fit(list(train_data[f"features_{approach}"].values), list(train_data["label"].values))
            predictions = model.predict(list(test_data[f"features_{approach}"].values))
            classification_evaluation = ClassificationEvaluation(
                y_true=list(test_data["label"].values), y_predicted=list(predictions)
            )
            accuracies[approach].append(classification_evaluation.accuracy)

    print("\nIndividual approach")
    print(f"Mean accuracy: {reduce(add, accuracies['individual']) / len(accuracies['individual'])}")
    print(f"Lilliefors test: {lilliefors_test(data=accuracies['individual'])}")
    print(
        f"One-sample T-test against random guessing: "
        f"{one_sample_t_test(sample=accuracies['individual'], population_mean=0.5, confidence_level=0.95)}"
    )

    print("\nSector approach")
    print(f"Mean accuracy: {reduce(add, accuracies['sector']) / len(accuracies['sector'])}")
    print(f"Lilliefors test: {lilliefors_test(data=accuracies['sector'])}")
    print(
        f"One-sample T-test against random guessing: "
        f"{one_sample_t_test(sample=accuracies['sector'], population_mean=0.5, confidence_level=0.95)}"
    )

    two_sample_t_test_results = two_sample_t_test(
        sample_1=accuracies["individual"], sample_2=accuracies["sector"], confidence_level=0.95
    )
    print(f"\nTwo-sample T-test between Individual and Sector approaches: {two_sample_t_test_results}")
