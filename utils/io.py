"""Collection of input/output utilities."""

import os

import pandas as pd


def save_pickle(data: pd.Series, directory: str, model: str):
    """Save a pandas Series as a pickle.

    Parameters:
        data: the pandas Series to save
        directory: the checkpoint directory
        model: the name of the model
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, model + '.pkl')
    data.to_pickle(path)


def load_pickle(directory: str, model: str) -> pd.Series:
    """Load a pickled object as a pandas Series.

    Parameters:
        directory: the checkpoint directory
        model: the name of the model

    Returns:
        the pandas Series
    """
    path = os.path.join(directory, model + '.pkl')
    return pd.read_pickle(path)


def save_csv(data: pd.Series, directory: str, model: str):
    """Save a pandas Series as a CSV file.

    Parameters:
        data: the pandas Series to save
        directory: the checkpoint directory
        model: the name of the model
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, model + '.csv')
    data.to_csv(path)


def load_csv(directory: str, model: str) -> pd.Series:
    """Load a CSV file as a pandas Series.

    Parameters:
        directory: the checkpoint directory
        model: the name of the model

    Returns:
        the pandas Series
    """
    path = os.path.join(directory, model + '.csv')
    return pd.read_csv(path)
