"""Tools for preprocessing the dataset."""

import argparse
from typing import Tuple

import geopandas as gpd
import numpy as np

from .filter import filter_crust_type, filter_nans
from .map import boundary_to_thickness, groupby_plate, merge_plates
from .reduce import ablation_study, reduce_attributes


def preprocess(
    data: gpd.GeoDataFrame, plate: gpd.GeoDataFrame, args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    data = merge_plates(data)

    # Separate X from y
    X = data
    y = -data["boundary topograpy", "upper crystalline crust"]
    geom = data["geom"]
    groups = data["plate index"]

    # Transform and reduce data attributes
    X = boundary_to_thickness(X)
    X = reduce_attributes(X)
    X = ablation_study(X, args.ablation)

    return X.values, y.values, geom.values, groups.values
