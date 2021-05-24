"""Preprocessing utilities that map data attributes.

This process is known as data transformation.
"""

from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def groupby_plate(data: gpd.GeoDataFrame, plate: gpd.GeoDataFrame) -> pd.DataFrame:
    """Group dataset by tectonic plate.

    Parameters:
        data: crust data
        plate: plate boundaries

    Returns:
        the dataset grouped by plate
    """
    # Flatten multi-index
    # https://github.com/geopandas/geopandas/issues/1764
    data.columns = data.columns.to_flat_index()
    data = data.set_geometry(("geom", ""))

    # Perform spatial join
    combined = gpd.sjoin(data, plate, how="inner", op="within")

    # Reconstruct multi-index
    combined = combined.rename(
        columns={
            "index_right": ("plate number", ""),
            "LAYER": ("layer", ""),
            "Code": ("code", ""),
            "PlateName": ("plate name", ""),
        }
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    # print(combined.value_counts(["plate number", "code", "plate name"]))

    return combined


def merge_plates(data: pd.DataFrame) -> pd.DataFrame:
    """Merge microplates and minor plates into nearby major plates.

    https://en.wikipedia.org/wiki/List_of_tectonic_plates

    Parameters:
        data: the entire dataset

    Returns:
        the modified dataset
    """
    all_to_subset = np.array(
        [
            0,  # Africa
            1,  # Antarctica
            0,  # Somalia -> Africa
            4,  # India -> Australia
            4,  # Australia
            5,  # Eurasia
            6,  # North America
            7,  # South America
            7,  # Nazca -> South America
            9,  # Pacific
            0,  # Arabia -> Africa
            5,  # Sunda -> Eurasia
            5,  # Timor -> Eurasia
            4,  # Kermadec -> Australia
            4,  # Kermadec -> Australia
            4,  # Tonga -> Australia
            4,  # Niuafo'ou -> Australia
            4,  # Woodlark -> Australia
            4,  # Maoke -> Australia
            4,  # South Bismarck -> Australia
            4,  # Solomon Sea -> Australia
            4,  # North Bismarck -> Australia
            4,  # New Hebrides -> Australia
            6,  # Caribbean -> North America
            7,  # Cocos -> South America
            5,  # Okhotsk -> Eurasia
            6,  # Juan de Fuca -> North America
            7,  # Altiplano -> South America
            7,  # North Andes -> South America
            5,  # Okinawa -> Eurasia
            5,  # Philippine Sea -> Eurasia
            5,  # Amur -> Eurasia
            4,  # Caroline -> Australia
            5,  # Mariana -> Eurasia
            4,  # Futuna -> Australia
            7,  # Scotia -> South America
            7,  # Shetland -> South America
            5,  # Aegean Sea -> Eurasia
            5,  # Anatolia -> Eurasia
            5,  # Yangtze -> Eurasia
            4,  # Burma -> Australia
            6,  # Rivera -> North America
            5,  # Birds Head -> Eurasia
            5,  # Molucca Sea -> Eurasia
            5,  # Banda Sea -> Eurasia
            4,  # Manus -> Australia
            4,  # Conway Reef -> Australia
            4,  # Balmoral Reef -> Australia
            4,  # Balmoral Reef -> Australia
            7,  # Easter -> South America
            7,  # Juan Fernandez -> South America
            7,  # Galapagos -> South America
            7,  # Sandwich -> South America
            6,  # Panama -> North America
        ]
    )

    data["plate number"] = all_to_subset[data["plate number"]]

    # print(data.value_counts(["plate number"]))

    return data


def boundary_to_thickness(data: pd.DataFrame) -> pd.DataFrame:
    """Convert boundary topography to layer thickness.

    Parameters:
        data: the entire dataset

    Returns:
        the modified dataset
    """
    bnds = data.pop("boundary topograpy")
    thickness = bnds.diff(periods=-1, axis=1)
    thickness = pd.concat([thickness], axis=1, keys=["thickness"], sort=False)
    return pd.concat([thickness, data], axis=1, sort=False)


def standardize(
    train: Union[pd.DataFrame, pd.Series],
    val: Union[pd.DataFrame, pd.Series],
    test: Union[pd.DataFrame, pd.Series],
) -> Tuple[
    Union[pd.DataFrame, pd.Series],
    Union[pd.DataFrame, pd.Series],
    Union[pd.DataFrame, pd.Series],
    StandardScaler,
]:
    """Standardize the dataset by subtracting the mean and dividing by the
    standard deviation.

    Parameters:
        train: training data
        val: validation data
        test: testing data

    Returns:
        standardized training data
        standardized validation data
        standardized testing data
        standardization scaler
    """
    is_series = isinstance(train, pd.Series)

    if is_series:
        # Must be a 2D array, pd.Series won't work
        train = pd.DataFrame(train)
        val = pd.DataFrame(val)
        test = pd.DataFrame(test)

    scaler = StandardScaler()

    # Compute the mean and std dev of the training set
    scaler.fit(train)

    # Transform the train/val/test sets
    arr_train = scaler.transform(train)
    arr_val = scaler.transform(val)
    arr_test = scaler.transform(test)

    if is_series:
        # Convert back to pd.Series
        train = pd.Series(arr_train.flatten(), index=train.index)
        val = pd.Series(arr_val.flatten(), index=val.index)
        test = pd.Series(arr_test.flatten(), index=test.index)
    else:
        # Convert back to pd.DataFrame
        train = pd.DataFrame(arr_train, index=train.index)
        val = pd.DataFrame(arr_val, index=val.index)
        test = pd.DataFrame(arr_test, index=test.index)

    return train, val, test, scaler


def inverse_standardize(
    train: pd.Series, val: pd.Series, test: pd.Series, scaler: StandardScaler
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Scale back the predictions to the original representation.

    Parameters:
        train: the training predictions
        val: the validation predictions
        test: the testing predictions
        scaler: the standardization scaler

    Returns:
        the scaled training predictions
        the scaled validation predictions
        the scaled testing predictions
    """
    if scaler is None:
        return train, val, test

    arr_train = scaler.inverse_transform(train)
    arr_val = scaler.inverse_transform(val)
    arr_test = scaler.inverse_transform(test)

    train = pd.Series(arr_train, index=train.index)
    val = pd.Series(arr_val, index=val.index)
    test = pd.Series(arr_test, index=test.index)

    return train, val, test
