"""Preprocessing utilities that map data attributes.

This process is known as data transformation.
"""

import argparse
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def spatial_join(crust: gpd.GeoDataFrame, age: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join two datasets.

    Parameters:
        crust: crust data
        age: seafloor age

    Returns:
        a single joined dataset
    """
    # Flatten multi-index
    # https://github.com/geopandas/geopandas/issues/1764
    crust.columns = crust.columns.to_flat_index()
    crust = crust.set_geometry(("geom", ""))

    # Perform spatial join
    combined = gpd.sjoin(crust, age, how="inner", op="intersects")
    combined = combined.drop(columns=["index_right"])

    # Reconstruct multi-index
    combined = combined.rename(
        columns={
            "age": ("age", ""),
        }
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    return combined


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
            "index_right": ("plate index", ""),
            "LAYER": ("layer", ""),
            "Code": ("code", ""),
            "PlateName": ("plate name", ""),
        }
    )
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    # print(combined.value_counts(["plate index", "code", "plate name"]))

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

    data["plate index"] = all_to_subset[data["plate index"]]

    # print(data.value_counts(["plate index"]))

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
    test: Union[pd.DataFrame, pd.Series],
    args: argparse.Namespace,
) -> Tuple[
    Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series], StandardScaler
]:
    """Standardize the dataset by subtracting the mean and dividing by the
    standard deviation.

    Parameters:
        train: training data
        test: testing data
        args: command-line arguments

    Returns:
        standardized training data
        standardized testing data
        standardization scaler
    """
    if args.model not in ["linear", "svr", "mlp"]:
        return train, test, None

    is_series = isinstance(train, pd.Series)

    if is_series:
        # Must be a 2D array, pd.Series won't work
        train = train.to_frame()
        test = test.to_frame()

    # Compute the mean and std dev of the training set
    scaler = StandardScaler()
    scaler.fit(train)

    # Transform the train/test sets
    train_arr = scaler.transform(train)
    test_arr = scaler.transform(test)

    if is_series:
        # Convert back to pd.Series
        train = pd.Series(train_arr.flatten(), index=train.index)
        test = pd.Series(test_arr.flatten(), index=test.index)
    else:
        # Convert back to pd.DataFrame
        train = pd.DataFrame(train_arr, index=train.index, columns=train.columns)
        test = pd.DataFrame(test_arr, index=test.index, columns=test.columns)

    return train, test, scaler


def inverse_standardize(
    test: pd.Series, predict: pd.Series, scaler: StandardScaler
) -> pd.Series:
    """Scale the predictions back to the original representation.

    Parameters:
        test: testing data
        predict: predicted data
        scaler: the standardization scaler

    Returns:
        the scaled testing predictions
    """
    if scaler is None:
        return test, predict

    test_arr = scaler.inverse_transform(test)
    predict_arr = scaler.inverse_transform(predict)

    test = pd.Series(test_arr, index=test.index)
    predict = pd.Series(predict_arr, index=predict.index)

    return test, predict
