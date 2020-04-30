"""Data loader for CRUST 1.0 dataset.

https://igppweb.ucsd.edu/~gabi/crust1.html
"""

import os

import numpy as np
import pandas as pd


def read_data(data_dir: str) -> pd.DataFrame:
    """Read CRUST 1.0 dataset.

    Parameters:
        data_dir: directory containing dataset

    Returns:
        the data
    """
    assert os.path.isdir(data_dir)

    kwargs = {
        'widths': [7] * 9,
        'header': None,
        'names': [
            'water', 'ice', 'upper sediments', 'middle sediments',
            'lower sediments', 'upper crystalline crust',
            'middle crystalline crust', 'lower crystalline crust', 'moho'
        ],
        'dtype': np.float32,
    }

    # Boundary topography
    # These files come with the CRUST 1.0 base model
    bnds = pd.read_fwf(os.path.join(data_dir, 'crust1.bnds'), **kwargs)

    # P/S-wave velocity
    kwargs['widths'] = [6] * 9
    vp = pd.read_fwf(os.path.join(data_dir, 'crust1.vp'), **kwargs)
    vs = pd.read_fwf(os.path.join(data_dir, 'crust1.vs'), **kwargs)

    # Density
    rho = pd.read_fwf(os.path.join(data_dir, 'crust1.rho'), **kwargs)

    # Crust type
    # This file comes with the CRUST 1.0 add-on
    ctype = np.loadtxt(
        os.path.join(data_dir, 'CNtype1-1.txt'), dtype=np.object)
    ctype = ctype.flatten()
    ctype = pd.DataFrame(ctype, columns=['crust type'])

    # Age
    # This file is downsampled from EarthByte
    # TODO: directly use EarthByte dataset with a different data loader
    kwargs.pop('widths')
    kwargs['names'] = ['longitude', 'latitude', 'age']
    age = pd.read_table(os.path.join(data_dir, 'age1.txt'), **kwargs)

    # Combine data
    data = pd.concat([bnds, vp, vs, rho, ctype, age], axis=1, keys=[
        'boundary topograpy', 'p-wave velocity', 's-wave velocity',
        'density', 'crust type', 'age'], sort=False)
    data.set_index([('age', 'latitude'), ('age', 'longitude')], inplace=True)
    data.index.rename(['latitude', 'longitude'], inplace=True)

    return data
