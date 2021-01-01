"""Tools for preprocessing the dataset."""

import argparse
from typing import Tuple

import pandas as pd

from .filter import filter_nans, filter_crust_type
from .reduce import ablation_study, reduce_attributes
from .transform import boundary_to_thickness


def preprocess(
    data: pd.DataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the dataset.

    Parameters:
        data: the entire dataset
        args: the command-line arguments

    Returns:
        a subset of the dataset
    """
    assert isinstance(data, pd.DataFrame)

    # Filter out data we don't want to train on
    data = filter_nans(data)
    data = filter_crust_type(data)

    # Separate X from y
    X, y = data, data["boundary topograpy", "upper crystalline crust"]
    y = -y

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)
    X = ablation_study(X, args.ablation)

    return X, y
