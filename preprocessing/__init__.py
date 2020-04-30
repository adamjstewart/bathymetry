"""Tools for preprocessing the dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split

from .filter import filter_nans, filter_crust_type
from .reduce import reduce_attributes
from .transform import boundary_to_thickness, standardize


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
    y = pd.DataFrame(y)

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)

    # Split train-validation-test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Standardize
    X_train, X_val, X_test, _ = standardize(X_train, X_val, X_test)
    y_train, y_val, y_test, y_scaler = standardize(y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, y_scaler


def train_val_test_split(X: pd.DataFrame, y: pd.DataFrame,
                         train_size: int = 60,
                         val_size: int = 20,
                         test_size: int = 20) -> tuple:
    """Split dataset into train/validation/test subsets.

    Parameters:
        X: input data
        y: ground truth labels
        train_size: percent of dataset for training
        val_size: percent of dataset for validation
        test_size: percent of dataset for testing

    Returns:
        train data
        validation data
        test data
        train labels
        validation labels
        test labels
    """
    assert len(X) == len(y)
    assert train_size + val_size + test_size == 100

    total_len = len(y)
    val_size *= total_len // 100
    test_size *= total_len // 100
    train_size = total_len - val_size - test_size

    assert train_size + val_size + test_size == total_len

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size)

    assert len(X_train) == train_size
    assert len(X_val) == val_size
    assert len(X_test) == test_size

    assert len(y_train) == train_size
    assert len(y_val) == val_size
    assert len(y_test) == test_size

    return X_train, X_val, X_test, y_train, y_val, y_test
