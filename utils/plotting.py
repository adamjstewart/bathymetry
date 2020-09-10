"""Collection of plotting utilities."""

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_world(data: pd.Series, title: str, legend: str):
    """Plot a world map with data.

    Parameters:
        data: the data to display
        title: the figure title
        legend: the legend label
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

    if 'difference' in legend:
        # Plotting the difference
        std = data.std()
        kwargs = {
            'cmap': cmocean.cm.balance,
            'vmin': -std,
            'vmax': +std,
        }
    elif 'bathymetry' in legend:
        # Plotting the absolute bathymetry
        kwargs = {
            'cmap': cmocean.cm.deep,
            'vmin': 0,
            'vmax': 10,
        }

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.coastlines()
    z = ax.pcolormesh(X, Y, C, transform=ccrs.PlateCarree(), **kwargs)
    cbar = fig.colorbar(z, ax=ax)
    ax.set_title(title)
    cbar.ax.set_ylabel(legend)
    plt.show()
