"""Collection of input/output utilities."""

import os
import pickle
from typing import Any

import pandas as pd


def save_pickle(data: Any, directory: str, filename: str):
    """Write the pickled representation of data to filename.

    Parameters:
        data: the data to save
        directory: the directory to save to
        filename: the filename to save to
    """
    path = os.path.join(directory, filename + '.pickle')
    print(f'Writing {path}...')
    os.makedirs(directory, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(directory: str, filename: str) -> Any:
    """Read the pickled representation of data from filename.

    Parameters:
        directory: the directory to load from
        filename: the filename to load from

    Returns:
        the original data
    """
    path = os.path.join(directory, filename + '.pickle')
    print(f'Reading {path}...')
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_csv(data: pd.Series, directory: str, filename: str):
    """Save a pandas Series as a CSV file.

    Parameters:
        data: the pandas Series to save
        directory: the directory to save to
        model: the filename to save to
    """
    path = os.path.join(directory, filename + '.csv')
    print(f'Writing {path}...')
    os.makedirs(directory, exist_ok=True)
    data.to_csv(path)


def load_csv(directory: str, filename: str) -> pd.Series:
    """Load a CSV file as a pandas Series.

    Parameters:
        directory: the directory to load from
        filename: the filename to load from

    Returns:
        the pandas Series
    """
    path = os.path.join(directory, filename + '.csv')
    print(f'Reading {path}...')
    return pd.read_csv(path)
