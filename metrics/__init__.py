"""Regression metrics for model evaluation."""

import math
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, r2_score


def evaluate(
    labels: pd.Series, predictions: pd.Series, sample_weight: pd.Series
) -> Dict[str, float]:
    """Prints the model performance evaluation.

    Args:
        labels: the ground truth labels
        predictions: the model predictions
        sample_weight: sample weights

    Returns:
        dictionary containing RMSE and R^2
    """
    rmse = root_mean_squared_error(labels, predictions, sample_weight=sample_weight)
    r2 = r2_score(labels, predictions, sample_weight=sample_weight)

    print("RMSE:", rmse)
    print("R^2: ", r2)

    return {"RMSE": rmse, "R^2": r2}


def weights(geom: pd.Series) -> pd.Series:
    """Calculate latitude-based weights for weighted loss functions and metrics.

    Args:
        geom: Geometry of the dataset

    Returns:
        Normalized weights for the dataset
    """
    weight = np.cos(np.radians(geom.y))
    return pd.Series(weight / np.mean(weight))
