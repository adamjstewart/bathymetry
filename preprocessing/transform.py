"""Preprocessing utilities that transform data attributes.

This process is known as data transformation.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


def boundary_to_thickness(data: pd.DataFrame) -> pd.DataFrame:
    """Convert boundary topography to layer thickness.

    Parameters:
        data: the entire dataset

    Returns:
        the modified dataset
    """
    bnds = data.pop('boundary topograpy')
    thickness = bnds.diff(periods=-1, axis=1)
    thickness = pd.concat([thickness], axis=1, keys=['thickness'], sort=False)
    return pd.concat([thickness, data], axis=1, sort=False)


def standardize(train: pd.DataFrame,
                val: pd.DataFrame,
                test: pd.DataFrame) -> tuple:
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
    scaler = StandardScaler()

    # Compute the mean and std dev of the training set
    scaler.fit(train)

    # Transform the train/val/test sets
    train = scaler.transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    return train, val, test, scaler
