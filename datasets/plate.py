"""Data loader for world tectonic plates and boundaries dataset.

https://github.com/fraxen/tectonicplates
"""

import os

import geopandas


def read_plate(data_dir: str) -> geopandas.GeoDataFrame:
    """Read world tectonic plates and boundaries dataset.

    Parameters:
        data_dir: directory containing datasets

    Returns:
        the data frame
    """
    path = os.path.join(data_dir, "tectonicplates-master", "PB2002_plates.shp")
    assert os.path.exists(path)

    print(f"Reading {path}...")
    return geopandas.read_file(path)
