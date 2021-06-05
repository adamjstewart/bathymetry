"""Data loader for various seafloor age datasets.

https://www.earthbyte.org/category/resources/data-models/seafloor-age/
"""

import os

import geopandas as gpd
import numpy as np

from utils.io import load_netcdf


def read_age(data_dir: str, year: int) -> gpd.GeoDataFrame:
    """Read seafloor age dataset.

    Parameters:
        data_dir: directory containing datasets
        year: year of data to use

    Returns:
        the data
    """
    data_dir = os.path.join(data_dir, f"age{year}")
    assert os.path.isdir(data_dir)

    # Load data
    filename = {
        2020: "age.2020.1.GTS2012.6m",
        2019: "Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid",
        2016: "Muller_etal_2016_AREPS_v1.17_PresentDay_AgeGrid",
        2013: "agegrid_0",
        2008: "age.3.6.unscaled",
    }
    ds = load_netcdf(data_dir, filename[year])

    # Downsample to 1-degree resolution
    lat = np.linspace(89.5, -89.5, 180)
    lon = np.linspace(179.5, -179.5, 360)
    ds = ds.reindex(lat=lat, lon=lon)

    # Convert to GeoDataFrame
    x, y = np.meshgrid(lon, lat)
    gs = gpd.points_from_xy(x.flatten(), y.flatten())
    gdf = gpd.GeoDataFrame(
        {"age": ds["z"].values.flatten()}, crs="EPSG:4326", geometry=gs
    )

    return gdf
