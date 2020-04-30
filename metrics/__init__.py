"""Regression metrics for model evaluation."""

import math

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(labels: np.ndarray, predictions: np.ndarray):
    """Prints the model performance evaluation.

    Parameters:
        labels: the ground truth labels
        predictions: the model predictions
    """
    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(labels, predictions)

    print('RMSE:', rmse)
    print('R^2: ', r2)
