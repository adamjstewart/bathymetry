"""Regression metrics for model evaluation."""

import math

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(labels: pd.Series, predictions: pd.Series):
    """Prints the model performance evaluation.

    Parameters:
        labels: the ground truth labels
        predictions: the model predictions
    """
    assert isinstance(labels, pd.Series)
    assert isinstance(predictions, pd.Series)

    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(labels, predictions)

    print('RMSE:', rmse)
    print('R^2: ', r2)
