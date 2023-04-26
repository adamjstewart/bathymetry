"""Preprocessing utilities that reduce the number of data attributes.

This process is known as data reduction.
"""

import geopandas as gpd
import pandas as pd

from .map import SUBPLATE_TO_SUPERPLATE, SUPERPLATE_TO_NAME


def reduce_attributes(data: pd.DataFrame) -> pd.DataFrame:
    """Remove attributes that we don't want to train on.

    Args:
        data: the entire dataset

    Returns:
        a subset of the dataset
    """
    labels = [
        ("thickness", "water"),  # keeping this would be cheating
        ("thickness", "moho"),  # this is all NaNs after the pd.diff
        ("crust type", "crust type"),
        ("geom", ""),
        ("plate index", ""),
        ("layer", ""),
        ("code", ""),
        ("plate name", ""),
    ]
    return data.drop(columns=labels)


def ablation_study(data: pd.DataFrame, labels: str) -> pd.DataFrame:
    """Remove attributes to perform an ablation study.

    Args:
        data: the entire dataset
        labels: sections to drop

    Returns:
        a subset of the dataset
    """
    if labels is not None:
        for label in labels.split(","):
            if label in [
                "thickness",
                "p-wave velocity",
                "s-wave velocity",
                "density",
                "age",
            ]:
                level = 0
            else:
                level = 1
            data = data.drop(columns=label, level=level)

    return data


def merge_plates(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge smaller plates into larger plates.

    Args:
        data: all tectonic plates

    Returns:
        only the largest tectonic plates
    """
    data["superplate"] = SUBPLATE_TO_SUPERPLATE
    data["superplate"] = data["superplate"].replace(SUPERPLATE_TO_NAME)
    plates = []
    names = list(SUPERPLATE_TO_NAME.values())
    for name in names:
        polygons = data.loc[data["superplate"] == name]
        plates.append(polygons["geometry"].unary_union)
    return gpd.GeoDataFrame({"name": names, "geometry": plates})
