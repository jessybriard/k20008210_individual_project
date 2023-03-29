"""Method(s) to compare the performance of individual and sector approach for a pair of forex ticker and commodities
ticker(s), and a choice of Machine Learning model. The method uses Monte-Carlo Cross-Validation to estimate the
performance of hourly predictions for both approaches. Then, for both the 'individual' and the 'sector' approach,
hypothesis testing is conducted on the results aggregated from these samples."""

import time
from datetime import timedelta
from functools import reduce
from operator import add
from typing import List

from numpy import mean, std
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

from src.tools.constants import PriceAttribute
from src.tools.hypothesis_testing import lilliefors_test, one_sample_t_test, two_sample_t_test
from src.tools.labeled_data_builder.monte_carlo_cross_validation import generate_train_test_sample
from src.tools.labeled_data_builder.time_series_forecasting import create_labeled_data
from src.tools.statistical_evaluation import ClassificationEvaluation
from src.tools.yfinance_data_provider import YfinanceDataProvider


def evaluate_and_compare_classification(
    forex_ticker: str, comdty_tickers: List[str], model, use_close_high_low: bool = False, nb_samples: int = 100
) -> None:
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
        use_close_high_low (bool): Whether to use hourly changes from the 'High' and 'Low' columns into the features, in
            addition to the 'Close' data.
        nb_samples (int): The number of samples to generate from the labeled data, using Monte-Carlo Cross-Validation.

    """

    features_length = 5
    attributes = (
        [PriceAttribute.CLOSE, PriceAttribute.HIGH, PriceAttribute.LOW]
        if use_close_high_low
        else [PriceAttribute.CLOSE]
    )
    data = YfinanceDataProvider.get_hourly_changes(attributes=attributes, tickers=comdty_tickers + [forex_ticker])
    labeled_data = create_labeled_data(
        attribute_label=PriceAttribute.CLOSE,
        ticker_label=forex_ticker,
        tickers_features=comdty_tickers + [forex_ticker],
        data=data,
        features_length=features_length,
    )

    start_time = time.time()
    accuracies = {"individual": [], "sector": []}
    for i in range(nb_samples):
        print(f"\n{i}")
        train_data, test_data = generate_train_test_sample(data=labeled_data, train_percentage=0.8)
        for approach in ["individual", "sector"]:
            model.fit(list(train_data[f"features_{approach}"].values), list(train_data["label_classification"].values))
            predictions = model.predict(list(test_data[f"features_{approach}"].values))
            classification_evaluation = ClassificationEvaluation(
                y_true=list(test_data["label_classification"].values), y_predicted=list(predictions)
            )
            accuracies[approach].append(classification_evaluation.accuracy)
            print(f"{approach}: {classification_evaluation.accuracy}")
    end_time = time.time()
    print(f"\nDuration: {timedelta(seconds=end_time - start_time)}")

    print("\nIndividual approach")
    print(f"Mean accuracy: {reduce(add, accuracies['individual']) / len(accuracies['individual'])}")
    print(f"Standard deviation (accuracy): {std(accuracies['individual'])}")
    print(f"Lilliefors test: {lilliefors_test(data=accuracies['individual'])}")
    print(
        f"One-sample T-test against random guessing: "
        f"{one_sample_t_test(sample=accuracies['individual'], population_mean=0.5, confidence_level=0.95)}"
    )

    print("\nSector approach")
    print(f"Mean accuracy: {reduce(add, accuracies['sector']) / len(accuracies['sector'])}")
    print(f"Standard deviation (accuracy): {std(accuracies['sector'])}")
    print(f"Lilliefors test: {lilliefors_test(data=accuracies['sector'])}")
    print(
        f"One-sample T-test against random guessing: "
        f"{one_sample_t_test(sample=accuracies['sector'], population_mean=0.5, confidence_level=0.95)}"
    )

    two_sample_t_test_results = two_sample_t_test(
        sample_1=accuracies["individual"], sample_2=accuracies["sector"], confidence_level=0.95
    )
    print(f"\nTwo-sample T-test between Individual and Sector approaches: {two_sample_t_test_results}")


def evaluate_and_compare_regression(
    attribute: PriceAttribute,
    forex_ticker: str,
    comdty_tickers: List[str],
    model,
    use_close_high_low: bool = False,
    nb_samples: int = 100,
) -> None:
    """Compare the performance of individual and sector approach for a pair of forex ticker and commodities
    ticker(s), and a choice of Regression model, for a selected price attribute ('Close', 'High', 'Low'). The method
    uses Monte-Carlo Cross-Validation to estimate the Mean-Absolute-Error of predicting (continous) hourly changes of
    the selected price attribute for both approaches. Then, for both the 'individual' and the 'sector' approach, using
    the results from these samples, hypothesis testing is conducted. First, the Lilliefors test is conducted to
    determine if the results come from a Gaussian (normal) distribution, at 95% confidence. Then, two-sample T-test is
    conducted to determine if the difference in Mean-Absolute-Error between the 'individual' and 'sector' approaches is
    statistically significant, at 95% confidence.

    Args:
        attribute (PriceAttribute): The price attribute to predict using Regression.
        forex_ticker (str): The ticker for the foreign exchange asset we want to predict for.
        comdty_tickers (List[str]): The ticker(s) for the commodities asset(s) we want to use as features to predict the
            foreign exchange asset.
        model: Instance of a scikit-learn Regression model, supporting methods fit() and predict().
        use_close_high_low (bool): Whether to use hourly changes from all 'Close', 'High' and 'Low' columns into the
            features, instead of the only predicted attribute.
        nb_samples (int): The number of samples to generate from the labeled data, using Monte-Carlo Cross-Validation.

    """

    features_length = 5
    attributes = [PriceAttribute.CLOSE, PriceAttribute.HIGH, PriceAttribute.LOW] if use_close_high_low else [attribute]
    data = YfinanceDataProvider.get_hourly_changes(attributes=attributes, tickers=comdty_tickers + [forex_ticker])
    labeled_data = create_labeled_data(
        attribute_label=attribute,
        ticker_label=forex_ticker,
        tickers_features=comdty_tickers + [forex_ticker],
        data=data,
        features_length=features_length,
    )

    start_time = time.time()
    baseline_model = DummyRegressor(strategy="mean")
    errors = {"individual": [], "sector": [], "baseline": []}
    for i in range(nb_samples):
        print(f"\n{i}")
        train_data, test_data = generate_train_test_sample(data=labeled_data, train_percentage=0.8)
        for approach in ["individual", "sector"]:
            model.fit(list(train_data[f"features_{approach}"].values), list(train_data["label_regression"].values))
            predictions = model.predict(list(test_data[f"features_{approach}"].values))
            errors[approach].append(mean_absolute_error(y_true=list(test_data["label_regression"]), y_pred=predictions))
            print(f"{approach}: {errors[approach][-1]}")
        baseline_model.fit(list(train_data["features_individual"].values), list(train_data["label_regression"].values))
        predictions = baseline_model.predict(list(test_data["features_individual"].values))
        errors["baseline"].append(mean_absolute_error(y_true=list(test_data["label_regression"]), y_pred=predictions))
        print(f"baseline: {errors['baseline'][-1]}")
    end_time = time.time()
    print(f"\nDuration: {timedelta(seconds=end_time - start_time)}")

    print(f"\nBaseline mean MAE: {mean(errors['baseline'])}")

    print("\nIndividual approach")
    print(f"Mean MAE: {reduce(add, errors['individual']) / len(errors['individual'])}")
    print(f"Standard deviation (MAE): {std(errors['individual'])}")
    print(f"Lilliefors test: {lilliefors_test(data=errors['individual'])}")
    two_sample_t_test_individual_baseline = two_sample_t_test(
        sample_1=errors["individual"], sample_2=errors["baseline"], confidence_level=0.95
    )
    print(f"Two-sample T-test against Baseline: {two_sample_t_test_individual_baseline}")

    print("\nSector approach")
    print(f"Mean MAE: {reduce(add, errors['sector']) / len(errors['sector'])}")
    print(f"Standard deviation (MAE): {std(errors['sector'])}")
    print(f"Lilliefors test: {lilliefors_test(data=errors['sector'])}")
    two_sample_t_test_sector_baseline = two_sample_t_test(
        sample_1=errors["sector"], sample_2=errors["baseline"], confidence_level=0.95
    )
    print(f"Two-sample T-test against Baseline: {two_sample_t_test_sector_baseline}")

    two_sample_t_test_results = two_sample_t_test(
        sample_1=errors["individual"], sample_2=errors["sector"], confidence_level=0.95
    )
    print(f"\nTwo-sample T-test between Individual and Sector approaches: {two_sample_t_test_results}")
