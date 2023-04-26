"""Regression metrics for model evaluation."""

import math
from typing import Dict

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def evaluate(labels: pd.Series, predictions: pd.Series) -> Dict[str, float]:
    """Prints the model performance evaluation.

    Args:
        labels: the ground truth labels
        predictions: the model predictions

    Returns:
        dictionary containing RMSE and R^2
    """
    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(labels, predictions)

    print("RMSE:", rmse)
    print("R^2: ", r2)

    return {"RMSE": rmse, "R^2": r2}
