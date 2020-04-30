"""Preprocessing utilities that transform data attributes.

This process is known as data transformation.
"""

import pandas as pd


def boundary_to_thickness(data: pd.DataFrame) -> pd.DataFrame:
    """Convert boundary topography to layer thickness.

    Parameters:
        data: the entire dataset

    Returns:
        the modified dataset
    """
    bnds = data.pop('boundary topograpy')
    thickness = bnds.diff(periods=-1, axis=1)
    thickness = pd.concat([thickness], axis=1, keys=['thickness'], sort=False)
    return pd.concat([thickness, data], axis=1, sort=False)
