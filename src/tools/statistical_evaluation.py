"""Statistical evaluation framework to assess the performance of prediction models."""

from typing import List


class ClassificationEvaluation:
    """Class Classification Evaluation.

    Methods to assess the performance of Classification prediction models. For our application, we only consider Boolean
    Classification models.
    """

    @staticmethod
    def accuracy(y_true: List[bool], y_predicted: List[bool]) -> float:
        """Calculate the accuracy of the prediction.

        Args:
            y_true (List[Any]): The true values for the predicted label.
            y_predicted (List[Any]): The predicted values for the label.

        Returns:
            accuracy (float): The value of the prediction's accuracy.
        """

        if len(y_true) != len(y_predicted):
            raise ValueError("Parameters 'y_true' and 'y_predicted' must be lists of same length.")

        return len([True for i in range(len(y_true)) if y_true[i] == y_predicted[i]]) / len(y_true)
