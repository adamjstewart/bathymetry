"""Preprocessing utilities that transform data attributes.

This process is known as data transformation.
"""

from typing import Tuple, Union

import pandas as pd
from sklearn.preprocessing import StandardScaler


def boundary_to_thickness(data: pd.DataFrame) -> pd.DataFrame:
    """Convert boundary topography to layer thickness.

    Parameters:
        data: the entire dataset

    Returns:
        the modified dataset
    """
    assert isinstance(data, pd.DataFrame)

    bnds = data.pop("boundary topograpy")
    thickness = bnds.diff(periods=-1, axis=1)
    thickness = pd.concat([thickness], axis=1, keys=["thickness"], sort=False)
    data = pd.concat([thickness, data], axis=1, sort=False)

    assert isinstance(data, pd.DataFrame)

    return data


def standardize(
    train: Union[pd.DataFrame, pd.Series],
    val: Union[pd.DataFrame, pd.Series],
    test: Union[pd.DataFrame, pd.Series],
) -> Tuple[
    Union[pd.DataFrame, pd.Series],
    Union[pd.DataFrame, pd.Series],
    Union[pd.DataFrame, pd.Series],
    StandardScaler,
]:
    """Standardize the dataset by subtracting the mean and dividing by the
    standard deviation.

    Parameters:
        train: training data
        val: validation data
        test: testing data

    Returns:
        standardized training data
        standardized validation data
        standardized testing data
        standardization scaler
    """
    assert isinstance(train, (pd.DataFrame, pd.Series))
    assert isinstance(val, (pd.DataFrame, pd.Series))
    assert isinstance(test, (pd.DataFrame, pd.Series))

    is_series = isinstance(train, pd.Series)

    if is_series:
        # Must be a 2D array, pd.Series won't work
        train = pd.DataFrame(train)
        val = pd.DataFrame(val)
        test = pd.DataFrame(test)

    scaler = StandardScaler()

    # Compute the mean and std dev of the training set
    scaler.fit(train)

    # Transform the train/val/test sets
    arr_train = scaler.transform(train)
    arr_val = scaler.transform(val)
    arr_test = scaler.transform(test)

    if is_series:
        # Convert back to pd.Series
        train = pd.Series(arr_train.flatten(), index=train.index)
        val = pd.Series(arr_val.flatten(), index=val.index)
        test = pd.Series(arr_test.flatten(), index=test.index)
    else:
        # Convert back to pd.DataFrame
        train = pd.DataFrame(arr_train, index=train.index)
        val = pd.DataFrame(arr_val, index=val.index)
        test = pd.DataFrame(arr_test, index=test.index)

    assert isinstance(train, (pd.DataFrame, pd.Series))
    assert isinstance(val, (pd.DataFrame, pd.Series))
    assert isinstance(test, (pd.DataFrame, pd.Series))

    return train, val, test, scaler


def inverse_standardize(
    train: pd.Series, val: pd.Series, test: pd.Series, scaler: StandardScaler
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Scale back the predictions to the original representation.

    Parameters:
        train: the training predictions
        val: the validation predictions
        test: the testing predictions
        scaler: the standardization scaler

    Returns:
        the scaled training predictions
        the scaled validation predictions
        the scaled testing predictions
    """
    assert isinstance(train, pd.Series)
    assert isinstance(val, pd.Series)
    assert isinstance(test, pd.Series)

    if scaler is None:
        return train, val, test

    arr_train = scaler.inverse_transform(train)
    arr_val = scaler.inverse_transform(val)
    arr_test = scaler.inverse_transform(test)

    train = pd.Series(arr_train, index=train.index)
    val = pd.Series(arr_val, index=val.index)
    test = pd.Series(arr_test, index=test.index)

    assert isinstance(train, pd.Series)
    assert isinstance(val, pd.Series)
    assert isinstance(test, pd.Series)

    return train, val, test
