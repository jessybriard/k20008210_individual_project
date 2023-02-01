"""Tests for classes in file hypothesis_testing.py."""

from unittest import TestCase

from src.tools.hypothesis_testing import lilliefors_test, one_sample_t_test


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

    # Tests for method one_sample_t_test()

    def test_one_sample_t_test_confidence_level_equals_zero(self):

        # Arrange
        sample = [0.51, 0.52, 0.5, 0.49, 0.51]
        population_mean = 0.5
        confidence_level = 0

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            one_sample_t_test(sample=sample, population_mean=population_mean, confidence_level=confidence_level)
        self.assertEqual("Confidence level for T-test must be between 0 and 1 (exclusive).", str(e.exception))

    def test_one_sample_t_test_confidence_level_equals_one(self):

        # Arrange
        sample = [0.51, 0.52, 0.5, 0.49, 0.51]
        population_mean = 0.5
        confidence_level = 1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            one_sample_t_test(sample=sample, population_mean=population_mean, confidence_level=confidence_level)
        self.assertEqual("Confidence level for T-test must be between 0 and 1 (exclusive).", str(e.exception))

    def test_one_sample_t_test_confidence_level_less_than_zero(self):

        # Arrange
        sample = [0.51, 0.52, 0.5, 0.49, 0.51]
        population_mean = 0.5
        confidence_level = -0.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            one_sample_t_test(sample=sample, population_mean=population_mean, confidence_level=confidence_level)
        self.assertEqual("Confidence level for T-test must be between 0 and 1 (exclusive).", str(e.exception))

    def test_one_sample_t_test_confidence_level_greater_than_one(self):

        # Arrange
        sample = [0.51, 0.52, 0.5, 0.49, 0.51]
        population_mean = 0.5
        confidence_level = 1.1

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            one_sample_t_test(sample=sample, population_mean=population_mean, confidence_level=confidence_level)
        self.assertEqual("Confidence level for T-test must be between 0 and 1 (exclusive).", str(e.exception))

    def test_one_sample_t_test_sample_is_empty(self):

        # Arrange
        sample = []
        population_mean = 0.5
        confidence_level = 0.95

        # Act / Assert
        with self.assertRaises(ValueError) as e:
            one_sample_t_test(sample=sample, population_mean=population_mean, confidence_level=confidence_level)
        self.assertEqual("Sample passed to one-sample T-test is empty.", str(e.exception))

    def test_one_sample_t_test_standard_deviation_is_zero(self):

        # Arrange
        sample = [0.51, 0.51, 0.51, 0.51, 0.51, 0.51]
        population_mean = 0.5
        confidence_level = 0.95

        # Act
        rejected_null_hypothesis, p_value = one_sample_t_test(
            sample=sample, population_mean=population_mean, confidence_level=confidence_level
        )

        # Assert
        self.assertTrue(rejected_null_hypothesis)
        self.assertEqual(0, p_value)

    def test_one_sample_t_test_does_not_reject_null_hypothesis(self):

        # Arrange
        sample = [0.51, 0.5, 0.49, 0.51, 0.51, 0.5]
        population_mean = 0.5
        confidence_level = 0.95

        # Act
        rejected_null_hypothesis, p_value = one_sample_t_test(
            sample=sample, population_mean=population_mean, confidence_level=confidence_level
        )

        # Assert
        self.assertFalse(rejected_null_hypothesis)
        self.assertAlmostEqual(0.136660839, p_value)

    def test_one_sample_t_test_rejects_null_hypothesis(self):

        # Arrange
        sample = [0.52, 0.52, 0.51, 0.53, 0.52, 0.5, 0.51, 0.525, 0.505, 0.52]
        population_mean = 0.5
        confidence_level = 0.95

        # Act
        rejected_null_hypothesis, p_value = one_sample_t_test(
            sample=sample, population_mean=population_mean, confidence_level=confidence_level
        )

        # Assert
        self.assertTrue(rejected_null_hypothesis)
        self.assertAlmostEqual(6.257997e-09, p_value)
