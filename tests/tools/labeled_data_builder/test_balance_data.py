"""Tests for methods in labeled_data_builder/balance_data.py."""

from collections import Counter
from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.balance_data import undersample


class TestBalanceData(TestCase):
    """Test class for methods in labeled_data_builder/balance_data.py."""

    # Tests for method undersample()

    def test_undersample_no_label_column_in_labeled_data(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
            }
        ).set_index("Date")

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            undersample(labeled_data=labeled_data)
        self.assertEqual("The given labeled_data does not contain a 'label' column.", str(e.exception))

    def test_undersample_multi_label_classification(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
                "label": [0, 1, 2, 0],
            }
        ).set_index("Date")

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            undersample(labeled_data=labeled_data)
        self.assertEqual(
            "The given labeled_data does not represent a binary classification problem (3 classes).", str(e.exception)
        )

    def test_undersample_empty_labeled_data(self):

        # Arrange
        labeled_data = pd.DataFrame(data={"Date": [], "features": [], "label": []}).set_index("Date")

        # Act
        balanced_data = undersample(labeled_data=labeled_data)

        # Assert
        expected_balanced_data = pd.DataFrame(data={"Date": [], "features": [], "label": []}).set_index("Date")
        self.assertTrue(expected_balanced_data.equals(balanced_data))

    def test_undersample_single_class(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": ["2022-11-09", "2022-11-10"],
                "features": [[0.008681, -0.032219], [-0.032219, -0.030936]],
                "label": [False, False],
            }
        ).set_index("Date")

        # Act
        balanced_data = undersample(labeled_data=labeled_data)

        # Assert
        expected_balanced_data = pd.DataFrame(
            data={
                "Date": ["2022-11-09", "2022-11-10"],
                "features": [[0.008681, -0.032219], [-0.032219, -0.030936]],
                "label": [False, False],
            }
        ).set_index("Date")
        self.assertTrue(expected_balanced_data.equals(balanced_data))

    def test_undersample_equal_distribution(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
                "label": [False, False, True, True],
            }
        ).set_index("Date")

        # Act
        balanced_data = undersample(labeled_data=labeled_data)

        # Assert
        expected_balanced_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
                "label": [False, False, True, True],
            }
        ).set_index("Date")
        self.assertTrue(expected_balanced_data.equals(balanced_data))

    def test_undersample_more_true(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
                "label": [True, False, True, True],
            }
        ).set_index("Date")

        # Act
        balanced_data = undersample(labeled_data=labeled_data)

        # Assert
        self.assertEqual({True: 1, False: 1}, dict(Counter(balanced_data["label"].values)))

    def test_undersample_more_false(self):

        # Arrange
        labeled_data = pd.DataFrame(
            data={
                "Date": [
                    "2022-11-09",
                    "2022-11-10",
                    "2022-11-11",
                    "2022-11-14",
                ],
                "features": [
                    [0.008681, -0.032219],
                    [-0.032219, -0.030936],
                    [-0.030936, 0.007222],
                    [0.007222, 0.031181],
                ],
                "label": [True, False, False, False],
            }
        ).set_index("Date")

        # Act
        balanced_data = undersample(labeled_data=labeled_data)

        # Assert
        self.assertEqual({True: 1, False: 1}, dict(Counter(balanced_data["label"].values)))
