"""Tools for preprocessing the dataset."""

import pandas as pd

from .filter import filter_nans, filter_crust_type
from .reduce import reduce_attributes
from .transform import boundary_to_thickness


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset.

    Parameters:
        data: the entire dataset

    Returns:
        a subset of the dataset
    """
    # Filter out data we don't want to train on
    data = filter_nans(data)
    data = filter_crust_type(data)

    # Separate X from y
    X, y = data, data['boundary topograpy', 'upper sediments']

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)

    # Split train-validation-test sets

    # Standardize
    # Must be trained on train and ran on val/test, so needs to happen after split
    # Need to keep Scaler for y so scaling can be inverted after prediction and before performance evaluation

    return X, y
