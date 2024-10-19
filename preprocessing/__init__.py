"""Tools for preprocessing the dataset."""

import argparse
from typing import Tuple

import geopandas as gpd
import pandas as pd

from .filter import filter_crust_type, filter_nans
from .map import boundary_to_thickness, groupby_grid, spatial_join
from .reduce import ablation_study, reduce_attributes


def preprocess(
    age: gpd.GeoDataFrame,
    crust: gpd.GeoDataFrame,
    plate: gpd.GeoDataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Preprocess the dataset.

    Args:
        age: seafloor age
        crust: crust data
        plate: plate boundaries
        args: the command-line arguments

    Returns:
        a subset of the dataset
    """
    # Join crust and age datasets
    data = spatial_join(crust, age)

    # Filter out data we don't want to train on
    data = filter_nans(data)
    data = filter_crust_type(data)

    # Group by grid
    data = groupby_grid(data, args.grid_size)

    # Separate X from y
    X = data
    y = -data["boundary topography", "upper crystalline crust"]
    geom = data["geom"]
    groups = data["grid cell"]

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)
    X = ablation_study(X, args.ablation)

    return X, y, geom, groups
