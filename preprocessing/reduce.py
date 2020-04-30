"""Preprocessing utilities that reduce the number of data attributes.

This process is known as data reduction.
"""

import pandas as pd


def reduce_attributes(data: pd.DataFrame) -> pd.DataFrame:
    """Remove attributes that we don't want to train on.

    Parameters:
        data: the entire dataset

    Returns:
        a subset of the dataset
    """
    labels = [
        ('thickness', 'water'),  # keeping this would be cheating
        ('thickness', 'moho'),   # this is all NaNs after the pd.diff
        ('crust type', 'crust type'),
    ]
    return data.drop(labels, axis=1)
