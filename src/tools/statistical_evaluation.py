"""Statistical evaluation framework to assess the performance of prediction models."""

from typing import List

from sklearn import metrics


class ClassificationEvaluation:
    """Class Classification Evaluation.

    Methods to assess the performance of Classification prediction models. For our application, we only consider Boolean
    Classification models.
    """

    accuracy, precision, recall, f1_score = 0, 0, 0, 0

    def __init__(self, y_true: List[bool], y_predicted: List[bool]) -> None:
        """Constructor for class ClassificationEvaluation. Check the length of the list parameters, compute the
        classification report from sklearn.methods, and extract relevant scores as instance variables.

        Args:
            y_true (List[bool]): The true values for the predicted label.
            y_predicted (List[bool]): The predicted values for the label.
        """
        self._validate_predictions_length(y_true=y_true, y_predicted=y_predicted)
        self.y_true = y_true
        self.y_predicted = y_predicted
        self.classification_report = metrics.classification_report(y_true=y_true, y_pred=y_predicted, output_dict=True)
        self._extract_scores_from_classification_report()

    @staticmethod
    def _validate_predictions_length(y_true: List, y_predicted: List) -> None:
        """Check that true values and predicted values are lists of same length, otherwise raise an Exception.

        Args:
            y_true (List[Any]): The true values for the predicted label.
            y_predicted (List[Any]): The predicted values for the label.
        """
        if not y_true or not y_predicted or len(y_true) != len(y_predicted):
            raise ValueError("Parameters 'y_true' and 'y_predicted' must be non-empty lists of same length.")

    def _extract_scores_from_classification_report(self) -> None:
        """Extract evaluation scores from the classification_report instance variable, retrieved from sklearn.metrics.
        Assign values for accuracy, precision, recall and f1_score.
        """
        self.accuracy = self.classification_report["accuracy"]
        weighted_avg_report: dict = self.classification_report["weighted avg"]
        self.precision = weighted_avg_report["precision"]
        self.recall = weighted_avg_report["recall"]
        self.f1_score = weighted_avg_report["f1-score"]

    def __str__(self) -> str:
        return (
            f"Accuracy: {self.accuracy}\n"
            f"Precision: {self.precision}\n"
            f"Recall: {self.recall}\n"
            f"F1 Score: {self.f1_score}"
        )
