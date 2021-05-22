"""Data loader for CRUST 1.0 dataset.

https://igppweb.ucsd.edu/~gabi/crust1.html
"""

import os

import geopandas
import numpy as np
import pandas as pd


def read_crust(data_dir: str) -> geopandas.GeoDataFrame:
    """Read CRUST 1.0 dataset.

    Parameters:
        data_dir: directory containing datasets

    Returns:
        the data
    """
    data_dir = os.path.join(data_dir, "CRUST1.0")
    assert os.path.isdir(data_dir)

    kwargs = {
        "widths": [7] * 9,
        "header": None,
        "names": [
            "water",
            "ice",
            "upper sediments",
            "middle sediments",
            "lower sediments",
            "upper crystalline crust",
            "middle crystalline crust",
            "lower crystalline crust",
            "moho",
        ],
        "dtype": np.float32,
    }

    # Boundary topography
    # These files come with the CRUST 1.0 base model
    fname = os.path.join(data_dir, "crust1.bnds")
    print(f"Reading {fname}...")
    bnds = pd.read_fwf(fname, **kwargs)

    # P/S-wave velocity
    kwargs["widths"] = [6] * 9
    fname = os.path.join(data_dir, "crust1.vp")
    print(f"Reading {fname}...")
    vp = pd.read_fwf(fname, **kwargs)

    fname = os.path.join(data_dir, "crust1.vs")
    print(f"Reading {fname}...")
    vs = pd.read_fwf(fname, **kwargs)

    # Density
    fname = os.path.join(data_dir, "crust1.rho")
    print(f"Reading {fname}...")
    rho = pd.read_fwf(fname, **kwargs)

    # Crust type
    # This file comes with the CRUST 1.0 add-on
    fname = os.path.join(data_dir, "CNtype1-1.txt")
    print(f"Reading {fname}...")
    ctype = np.loadtxt(fname, dtype=object)
    ctype = ctype.flatten()
    ctype = pd.DataFrame(ctype, columns=["crust type"])

    # Age
    # This file is downsampled from EarthByte
    # TODO: directly use EarthByte dataset with a different data loader
    kwargs.pop("widths")
    kwargs["names"] = ["longitude", "latitude", "age"]
    fname = os.path.join(data_dir, "age1.txt")
    print(f"Reading {fname}...")
    age = pd.read_table(fname, **kwargs)

    # Combine data
    data = pd.concat(
        [bnds, vp, vs, rho, ctype, age],
        axis=1,
        keys=[
            "boundary topograpy",
            "p-wave velocity",
            "s-wave velocity",
            "density",
            "crust type",
            "age",
        ],
        sort=False,
    )

    # Convert to GeoDataFrame
    # https://github.com/geopandas/geopandas/issues/1763
    data["geom"] = geopandas.points_from_xy(
        data["age", "longitude"], data["age", "latitude"]
    )
    data = geopandas.GeoDataFrame(
        data,
        crs="EPSG:4326",
        geometry="geom",
    )
    data.drop(columns=[("age", "longitude"), ("age", "latitude")], inplace=True)

    return data
