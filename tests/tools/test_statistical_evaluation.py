"""Tests for classes in file statistical_evaluation.py."""

from unittest import TestCase
from unittest.mock import patch

from src.tools.statistical_evaluation import ClassificationEvaluation


class TestClassificationEvaluation(TestCase):
    """Test class for methods in class ClassificationEvaluation."""

    def test_validate_predictions_length_empty_lists(self):

        # Arrange
        y_true = []
        y_predicted = []

        # Act / Arrange
        with self.assertRaises(ValueError) as e:
            ClassificationEvaluation._validate_predictions_length(y_true=y_true, y_predicted=y_predicted)
        self.assertEqual(
            str(e.exception), "Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length."
        )

    def test_validate_predictions_length_y_true_empty_list(self):

        # Arrange
        y_true = []
        y_predicted = [False, False, True, True]

        # Act / Arrange
        with self.assertRaises(ValueError) as e:
            ClassificationEvaluation._validate_predictions_length(y_true=y_true, y_predicted=y_predicted)
        self.assertEqual(
            str(e.exception), "Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length."
        )

    def test_validate_predictions_length_y_predicted_empty_list(self):

        # Arrange
        y_true = [True, False, True]
        y_predicted = []

        # Act / Arrange
        with self.assertRaises(ValueError) as e:
            ClassificationEvaluation._validate_predictions_length(y_true=y_true, y_predicted=y_predicted)
        self.assertEqual(
            str(e.exception), "Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length."
        )

    def test_validate_predictions_length_different_lengths(self):

        # Arrange
        y_true = [True, False, True]
        y_predicted = [False, False, True, True]

        # Act / Arrange
        with self.assertRaises(ValueError) as e:
            ClassificationEvaluation._validate_predictions_length(y_true=y_true, y_predicted=y_predicted)
        self.assertEqual(
            str(e.exception), "Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length."
        )

    def test_validate_predictions_length_same_lengths(self):

        # Arrange
        y_true = [True, False, True, True]
        y_predicted = [False, False, True, True]

        # Act / Assert
        self.assertIsNone(ClassificationEvaluation._validate_predictions_length(y_true=y_true, y_predicted=y_predicted))

    def test_constructor_calls_validate_predictions_length(self):

        # Arrange
        y_true = []
        y_predicted = []

        # Act / Assert
        with patch(
            "src.tools.statistical_evaluation.ClassificationEvaluation._validate_predictions_length",
            wraps=ClassificationEvaluation._validate_predictions_length,
        ) as mock_validate_predictions_length:
            with self.assertRaises(ValueError) as e:
                ClassificationEvaluation(y_true=y_true, y_predicted=y_predicted)
            mock_validate_predictions_length.assert_called()
            self.assertEqual(
                str(e.exception), "Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length."
            )

    @patch("sklearn.metrics.classification_report")
    def test_extract_scores_from_classification_report_returns_weighted_average_scores(
        self, mock_classification_report_method
    ):

        # Arrange
        y_true = [True, False, True, False, True]
        y_predicted = [False, False, False, True, False]
        classification_report = {
            "False": {"f1-score": 0.3333333333333333, "precision": 0.25, "recall": 0.5, "support": 2},
            "True": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3},
            "accuracy": 0.2,
            "macro avg": {"f1-score": 0.16666666666666666, "precision": 0.125, "recall": 0.25, "support": 5},
            "weighted avg": {"f1-score": 0.13333333333333333, "precision": 0.1, "recall": 0.2, "support": 5},
        }
        mock_classification_report_method.return_value = classification_report

        # Act
        evaluation = ClassificationEvaluation(y_true=y_true, y_predicted=y_predicted)

        # Assert
        self.assertEqual(evaluation.classification_report, classification_report)
        self.assertEqual(evaluation.accuracy, 0.2)
        self.assertEqual(evaluation.precision, 0.1)
        self.assertEqual(evaluation.recall, 0.2)
        self.assertEqual(evaluation.f1_score, 0.13333333333333333)

    @patch("sklearn.metrics.classification_report")
    def test_str_overload(self, mock_classification_report_method):

        # Arrange
        y_true = [True, False, True, False, True]
        y_predicted = [False, False, False, True, False]
        classification_report = {
            "False": {"f1-score": 0.3333333333333333, "precision": 0.25, "recall": 0.5, "support": 2},
            "True": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0, "support": 3},
            "accuracy": 0.2,
            "macro avg": {"f1-score": 0.16666666666666666, "precision": 0.125, "recall": 0.25, "support": 5},
            "weighted avg": {"f1-score": 0.13333333333333333, "precision": 0.1, "recall": 0.2, "support": 5},
        }
        mock_classification_report_method.return_value = classification_report

        # Act
        evaluation = ClassificationEvaluation(y_true=y_true, y_predicted=y_predicted)

        # Assert
        self.assertEqual(str(evaluation), "Accuracy: 0.2\nPrecision: 0.1\nRecall: 0.2\nF1 Score: 0.13333333333333333")
