"""Methods to conduct statistical hypothesis testing on observed data."""

from typing import List, Tuple

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
