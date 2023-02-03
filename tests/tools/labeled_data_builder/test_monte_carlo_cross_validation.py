"""Tests for methods in labeled_data_builder/monte_carlo_cross_validation.py."""

from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.monte_carlo_cross_validation import generate_train_test_sample


class TestLabeledDataBuilderMonteCarloCrossValidation(TestCase):
    """Test class for methods in labeled_data_builder/monte_carlo_cross_validation.py."""

    # Tests for method generate_train_test_sample()

    def test_generate_train_test_sample_train_percentage_less_than_zero(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        train_percentage = -0.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            generate_train_test_sample(data=data, train_percentage=train_percentage)
        self.assertEqual("Parameter 'train_percentage' must be a number between 0 and 1 inclusive.", str(e.exception))

    def test_generate_train_test_sample_train_percentage_greater_than_one(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        train_percentage = 1.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            generate_train_test_sample(data=data, train_percentage=train_percentage)
        self.assertEqual("Parameter 'train_percentage' must be a number between 0 and 1 inclusive.", str(e.exception))

    def test_generate_train_test_sample_train_percentage_equals_zero(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        train_percentage = 0

        # Act
        train_sample, test_sample = generate_train_test_sample(data=data, train_percentage=train_percentage)

        # Assert
        expected_test_sample = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        self.assertTrue(train_sample.empty)
        self.assertEqual(set(expected_test_sample.index), set(test_sample.index))

    def test_generate_train_test_sample_train_percentage_equals_one(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        train_percentage = 1

        # Act
        train_sample, test_sample = generate_train_test_sample(data=data, train_percentage=train_percentage)

        # Assert
        expected_train_sample = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        self.assertEqual(set(expected_train_sample.index), set(train_sample.index))
        self.assertTrue(test_sample.empty)

    def test_generate_train_test_sample_data_is_empty(self):

        # Arrange
        data = pd.DataFrame()
        train_percentage = 0.8

        # Act
        train_sample, test_sample = generate_train_test_sample(data=data, train_percentage=train_percentage)

        # Assert
        self.assertTrue(train_sample.empty)
        self.assertTrue(pd.DataFrame().equals(train_sample))
        self.assertTrue(test_sample.empty)
        self.assertTrue(pd.DataFrame().equals(test_sample))

    def test_generate_train_test_sample_sample_data_is_correct_length(self):

        # Arrange
        data = pd.DataFrame(
            data={
                "features": [
                    [0.1, -0.05],
                    [-0.05, 0.02],
                    [0.02, 0.12],
                    [0.12, -0.04],
                    [-0.04, -0.1],
                    [-0.1, 0.14],
                    [0.14, -0.12],
                    [-0.12, 0.2],
                    [0.2, 0.01],
                    [0.01, 0.13],
                    [0.13, -0.09],
                ],
                "label": [True, True, False, False, True, False, True, True, True, False, False],
            }
        )
        train_percentage = 0.8

        # Act
        train_sample, test_sample = generate_train_test_sample(data=data, train_percentage=train_percentage)

        # Assert
        self.assertEqual(9, len(train_sample))
        self.assertEqual(2, len(test_sample))
        self.assertEqual(11, len(set(list(train_sample.index) + list(test_sample.index))))
