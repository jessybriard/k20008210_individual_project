"""Tests for methods in labeled_data_builder/bootstrap.py."""

import random
from unittest import TestCase

import pandas as pd

from src.tools.labeled_data_builder.bootstrap import generate_sample


class TestLabeledDataBuilderBoostrap(TestCase):
    """Test class for methods in labeled_data_builder/bootstrap.py."""

    # Tests for method generate_sample()

    def test_generate_sample_data_is_empty(self):

        # Arrange
        data = pd.DataFrame(data={"features": [], "label": []})

        # Act
        sample = generate_sample(data=data)

        # Assert
        self.assertEqual(["features", "label"], list(sample.columns))
        self.assertTrue(sample.empty)

    def test_generate_sample_same_length_as_original_data(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )

        # Act
        sample = generate_sample(data=data)

        # Assert
        self.assertEqual(3, len(sample))

    def test_generate_sample_can_contain_duplicate_rows(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        random.seed(0)

        # Act
        sample = generate_sample(data=data)

        # Assert
        self.assertTrue(True in sample.index.duplicated())

    def test_generate_sample_can_create_different_samples(self):

        # Arrange
        data = pd.DataFrame(
            data={"features": [[0.1, -0.05], [-0.05, 0.02], [0.02, 0.12]], "label": [True, True, False]}
        )
        random.seed(0)

        # Act
        sample_1 = generate_sample(data=data)
        sample_2 = generate_sample(data=data)

        # Assert
        self.assertFalse(sample_1.equals(sample_2))
