"""Statistical evaluation framework to assess the performance of prediction models."""

from typing import List

from sklearn import metrics


class ClassificationEvaluation:
    """Class Classification Evaluation.

    Methods to assess the performance of Classification prediction models. For our application, we only consider Boolean
    Classification models.
    """

    def __init__(self, y_true: List[bool], y_predicted: List[bool]):
        self._validate_predictions_length(y_true, y_predicted)
        self.y_true = y_true
        self.y_predicted = y_predicted

    @staticmethod
    def _validate_predictions_length(y_true, y_predicted):
        """Check that true values and predicted values are lists of same length, otherwise raise an Exception.

        Args:
            y_true (List[Any]): The true values for the predicted label.
            y_predicted (List[Any]): The predicted values for the label.
        """
        if len(y_true) != len(y_predicted):
            raise ValueError("Parameters 'y_true' and 'y_predicted' must be lists of same length.")

    def accuracy(self) -> float:
        """Calculate the Accuracy of the prediction.

        Returns:
            accuracy (float): The value of the prediction's accuracy.
        """
        return len([True for i in range(len(self.y_true)) if self.y_true[i] == self.y_predicted[i]]) / len(self.y_true)

    def f1_score(self) -> dict:
        """Calculate the F1 Score of the prediction, for each possible label value.

        Returns:
            f1_scores (dict): The value of the prediction's F1 Score, for each possible label value.
        """
        labels_score = metrics.f1_score(y_true=self.y_true, y_pred=self.y_predicted, labels=[True, False], average=None)
        return {True: labels_score[0], False: labels_score[1]}
