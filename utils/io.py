"""Collection of input/output utilities."""

import argparse
import os
import pickle
from typing import Any, Dict

import pandas as pd
from sklearn.base import BaseEstimator


def save_pickle(data: Any, directory: str, filename: str) -> None:
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


def save_csv(data: pd.Series, directory: str, filename: str) -> None:
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


def save_checkpoint(
        model: BaseEstimator,
        args: argparse.Namespace,
        accuracies: Dict[str, Dict[str, float]],
) -> None:
    """Save a checkpoint for hyperparameter tuning.

    Parameters:
        model: the trained model
        args: the hyperparameters
        accuracies: the performance metrics
    """
    data = {
        'args': args,
        'accuracies': accuracies,
    }

    values = [value for (key, value) in sorted(model.get_params().items())]
    values = ['checkpoint', args.model] + values
    filename = '-'.join([str(value) for value in values])

    save_pickle(data, args.checkpoint_dir, filename)
