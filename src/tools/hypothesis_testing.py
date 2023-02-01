"""Methods to conduct statistical hypothesis testing on observed data."""

from math import sqrt
from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from statsmodels.stats.diagnostic import lilliefors


def lilliefors_test(data: List[float]) -> Tuple[bool, float]:
    """Conduct the Lilliefors test on the observed data to determine if the data comes from a Gaussian (normal)
    distribution. The lilliefors method from the underlying statsmodels library will calculate the associated p-value,
    we then reject the null hypothesis that the data comes from a Gaussian distribution if p-value < 0.05, at 95%
    confidence level. Otherwise, we determine that the data does come from a Gaussian distribution.

    Args:
        data (List[float]): The data for which we determine if it comes from a Gaussian distribution.

    Returns:
        normal_distribution (bool): True if the data comes from a Gaussian distribution, False otherwise.
        p_value (float): The p-value calculated through the Lilliefors test for this data.

    """

    if not data:
        raise ValueError("Data passed to Lilliefors test is empty.")

    ksstat, p_value = lilliefors(x=data, dist="norm")
    return p_value >= 0.05, p_value


def one_sample_t_test(
    sample: List[float], population_mean: float, confidence_level: float = 0.95
) -> Tuple[bool, float]:
    """Conduct a right-tailed one-sample T-test to determine if the difference between the mean of a sample and the
    theoretical mean of a population is statistically significant, where mean(sample) > population_mean. This is an
    instance of hypothesis testing where the null hypothesis is that the observed sample is not significantly different
    from the population. This test assumes that the sample and population come from Gaussian (normal) distributions. If
    the calculated p_value is lower than (1 - confidence_level), then we can reject the null hypothesis and determine
    that the observed sample is significantly different from the population.

    Args:
        sample (List[float]): The observed sample to test.
        population_mean (float): The theoretical mean of the population.
        confidence_level (float): The confidence level of the T-test.

    Returns:
        rejected_null_hypothesis (bool): True if the null hypothesis is rejected (p_value < (1 - confidence_level)).
        p_value (float): The p-value calculated through this one-sample T-test.

    """

    if not 0.5 < confidence_level < 1:
        raise ValueError("Confidence level for T-test must be between 0.5 and 1 (exclusive).")

    if not sample:
        raise ValueError("Sample passed to one-sample T-test is empty.")

    if len(set(sample)) == 1:  # Standard deviation is zero, we reject the null hypothesis
        return True, 0

    t_value = (np.mean(sample) - population_mean) / (np.std(sample) / sqrt(len(sample)))
    p_value = 1 - norm.cdf(t_value)
    return p_value < 1 - confidence_level, p_value


def two_sample_t_test(
    sample_1: List[float], sample_2: List[float], confidence_level: float = 0.95
) -> Tuple[bool, float]:
    """Conduct a right-tailed two-sample T-test to determine if the difference between the mean of two samples is
    statistically significant. This is an instance of hypothesis testing where the null hypothesis is that the two
    observed sample are not significantly different. This test assumes that the sample and population come from Gaussian
    (normal) distributions. If the calculated p_value is lower than (1 - confidence_level), then we can reject the null
    hypothesis and determine that the two observed samples are significantly different.

    Args:
        sample_1 (List[float]): The first observed sample to test.
        sample_2 (List[float]): The second observed sample to test.
        confidence_level (float): The confidence level of the T-test.

    Returns:
        rejected_null_hypothesis (bool): True if the null hypothesis is rejected (p_value < (1 - confidence_level)).
        p_value (float): The p-value calculated through this two-sample T-test.

    """

    if not 0.5 < confidence_level < 1:
        raise ValueError("Confidence level for T-test must be between 0.5 and 1 (exclusive).")

    if not sample_1 or not sample_2:
        raise ValueError("Samples passed to two-sample T-test must not be empty.")

    if len(set(sample_1)) == 1 and len(set(sample_2)) == 1:
        if set(sample_1) == set(sample_2):
            return False, 0.5
        return True, 0

    t_value = abs(np.mean(sample_1) - np.mean(sample_2)) / sqrt(
        (np.std(sample_1) / len(sample_1)) + (np.std(sample_2) / len(sample_2))
    )
    p_value = 1 - norm.cdf(t_value)
    return p_value < 1 - confidence_level, p_value
