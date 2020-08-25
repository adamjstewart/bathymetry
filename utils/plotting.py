"""Collection of plotting utilities."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_world(data: pd.Series):
    """Plot a world map with data.

    Parameters:
        data: the data to display
    """
    assert isinstance(data, pd.Series)

    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 180, 361)
    X, Y = np.meshgrid(lon, lat)

    C = np.full((180, 360), np.nan, dtype=np.float32)
    for i in range(180):
        for j in range(360):
            idx = (i - 89.5, j - 179.5)
            try:
                C[i, j] = data[idx]
            except KeyError:
                pass

    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines()
    z = ax.pcolormesh(X, Y, C, cmap='bwr', vmin=-1, vmax=1,
                      transform=ccrs.PlateCarree())
    fig.colorbar(z, ax=ax)
    plt.show()
