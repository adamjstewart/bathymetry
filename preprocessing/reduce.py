"""Preprocessing utilities that reduce the number of data attributes.

This process is known as data reduction.
"""

from typing import Sequence

import pandas as pd


def reduce_attributes(data: pd.DataFrame) -> pd.DataFrame:
    """Remove attributes that we don't want to train on.

    Parameters:
        data: the entire dataset

    Returns:
        a subset of the dataset
    """
    assert isinstance(data, pd.DataFrame)

    labels = [
        ('thickness', 'water'),  # keeping this would be cheating
        ('thickness', 'moho'),   # this is all NaNs after the pd.diff
        ('crust type', 'crust type'),
    ]
    data = data.drop(columns=labels)

    assert isinstance(data, pd.DataFrame)

    return data


def ablation_study(data: pd.DataFrame, labels: Sequence) -> pd.DataFrame:
    """Remove attributes to perform an ablation study.

    Parameters:
        data: the entire dataset
        labels: sections to drop

    Returns:
        a subset of the dataset
    """
    assert isinstance(data, pd.DataFrame)

    for label in labels.split(','):
        if label in [
            'thickness', 'p-wave velocity', 's-wave velocity', 'density', 'age'
        ]:
            level = 0
        else:
            level = 1
        data = data.drop(columns=label, level=level)

    assert isinstance(data, pd.DataFrame)

    return data
