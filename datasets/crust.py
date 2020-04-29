"""Data loader for CRUST 1.0 dataset.

https://igppweb.ucsd.edu/~gabi/crust1.html
"""

import os

import numpy as np
import pandas as pd


def read_data(data_dir):
    """Read CRUST 1.0 dataset.

    Parameters:
        data_dir (str): directory containing dataset

    Returns:
        pandas.DataFrame: the data
    """
    assert os.path.isdir(data_dir)

    kwargs = {
        'header': None,
        'names': [
            'water', 'ice', 'upper sediments', 'middle sediments',
            'lower sediments', 'upper crystalline crust',
            'middle crystalline crust', 'lower crystalline crust', 'moho'
        ],
        'dtype': np.float32,
    }

    # Boundary topography
    bnds = pd.read_fwf(os.path.join(data_dir, 'crust1.bnds'), **kwargs)

    # P/S-wave velocity
    vp = pd.read_fwf(os.path.join(data_dir, 'crust1.vp'), **kwargs)
    vs = pd.read_fwf(os.path.join(data_dir, 'crust1.vs'), **kwargs)

    # Density
    rho = pd.read_fwf(os.path.join(data_dir, 'crust1.rho'), **kwargs)

    # Crust type
    ctype = np.loadtxt(
        os.path.join(data_dir, 'CNtype1-1.txt'), dtype=np.object)
    ctype = ctype.flatten()
    ctype = pd.get_dummies(ctype, dtype=np.float32)  # one-hot encoding

    # Age
    kwargs['names'] = ['longitude', 'latitude', 'age']
    age = pd.read_table(os.path.join(data_dir, 'age1.txt'), **kwargs)

    # Bathymetry
    kwargs['names'] = ['longitude', 'latitude', 'bathymetry']
    bath = pd.read_table(os.path.join(data_dir, 'xyz-bd3_oprm'), **kwargs)

    # Combine data
    data = pd.concat([bnds, vp, vs, rho, ctype, age, bath], axis=1, keys=[
        'boundary topograpy', 'p-wave velocity', 's-wave velocity',
        'density', 'crust type', 'age', 'bathymetry'], sort=False)
    data.set_index([('age', 'latitude'), ('age', 'longitude')], inplace=True)
    data.drop([('bathymetry', 'longitude'), ('bathymetry', 'latitude')],
              axis=1, inplace=True)
    data.index.rename(['latitude', 'longitude'], inplace=True)

    assert isinstance(data, pd.DataFrame)

    return data
