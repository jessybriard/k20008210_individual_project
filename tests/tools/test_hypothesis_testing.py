"""Tests for classes in file hypothesis_testing.py."""

from unittest import TestCase

from src.tools.hypothesis_testing import lilliefors_test


class TestHypothesisTesting(TestCase):
    """Test class for methods in file hypothesis_testing.py."""

    # Tests for method lilliefors_test()

    def test_lilliefors_test_data_is_empty(self):

        # Arrange
        data = []

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            lilliefors_test(data=data)
        self.assertEqual("Data passed to Lilliefors test is empty.", str(e.exception))

    def test_lilliefors_test_data_has_less_than_four_observations(self):

        # Arrange
        data = [0, 1, 0]

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            lilliefors_test(data=data)
        self.assertEqual("Test for distribution norm requires at least 4 observations", str(e.exception))

    def test_lilliefors_test_data_does_not_come_from_normal_distribution(self):

        # Arrange
        data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        # Act
        normal_distribution, p_value = lilliefors_test(data=data)

        # Assert
        self.assertFalse(normal_distribution)
        self.assertTrue(p_value < 0.05)

    def test_lilliefors_test_data_comes_from_normal_distribution(self):

        # Arrange
        data = [0.4, 0.45, 0.5, 0.45, 0.4]

        # Act
        normal_distribution, p_value = lilliefors_test(data=data)

        # Assert
        self.assertTrue(normal_distribution)
        self.assertTrue(p_value >= 0.05)
