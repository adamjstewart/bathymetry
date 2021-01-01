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
    data = geopandas.read_file(path)

    # Some plates, like AU and PA, are coded as MULTIPOLYGONs,
    # while others, like KE and BR, have two separate POLYGON entries.
    # Merge these so that there is a single entry per plate.
    # https://geopandas.org/aggregation_with_dissolve.html
    data = data.dissolve(by="Code", as_index=False)

    return data
