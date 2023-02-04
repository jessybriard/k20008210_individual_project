"""Method(s) to balance labeled data."""

from collections import Counter
from random import randint

import pandas as pd


def undersample(labeled_data: pd.DataFrame) -> pd.DataFrame:
    """Balance a classification dataset using under-sampling. We assume binary classification.

    Args:
        labeled_data (pd.DataFrame): The original (unbalanced labeled data).

    Returns:
        balanced_data (pd.DataFrame): The modified balanced labeled data.

    """

    if "label_classification" not in labeled_data.columns:
        raise ValueError("The given labeled_data does not contain a 'label_classification' column.")

    class_distribution = Counter(labeled_data["label_classification"].values).most_common()

    if len(class_distribution) > 2:
        raise ValueError(
            f"The given labeled_data does not represent a binary classification problem "
            f"({len(class_distribution)} classes)."
        )
    elif len(class_distribution) < 2:
        return labeled_data

    while class_distribution[0][1] > class_distribution[1][1]:
        i = randint(0, len(labeled_data) - 1)
        if labeled_data.iloc[i]["label_classification"] == class_distribution[0][0]:
            labeled_data = labeled_data.drop(labeled_data.index[i])
            class_distribution = Counter(labeled_data["label_classification"].values).most_common()

    return labeled_data
