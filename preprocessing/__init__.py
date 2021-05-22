"""Tools for preprocessing the dataset."""

import argparse
from typing import Tuple

import geopandas as gpd
import pandas as pd

from .filter import filter_nans, filter_crust_type
from .reduce import ablation_study, reduce_attributes
from .map import boundary_to_thickness, groupby_plate


def preprocess(
    data: gpd.GeoDataFrame, plate: gpd.GeoDataFrame, args: argparse.Namespace
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Preprocess the dataset.

    Parameters:
        data: crust data
        plate: plate boundaries
        args: the command-line arguments

    Returns:
        a subset of the dataset
    """
    # Filter out data we don't want to train on
    data = filter_nans(data)
    data = filter_crust_type(data)

    # Group by tectonic plate
    data = groupby_plate(data, plate)

    # Separate X from y
    X, y, groups = (
        data,
        data["boundary topograpy", "upper crystalline crust"],
        data["index_right"],
    )
    y = -y

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)
    X = ablation_study(X, args.ablation)

    return X, y, groups
