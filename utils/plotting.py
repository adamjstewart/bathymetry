"""Collection of plotting utilities."""

import os

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


def plot_world(directory: str, data: pd.Series, title: str, legend: str) -> None:
    """Plot a world map with data.

    Parameters:
        directory: the directory to save the file to
        data: the data to display
        title: the figure title
        legend: the legend label
    """
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

    if "difference" in legend:
        # Plotting the difference
        std = data.std()
        kwargs = {"cmap": cmocean.cm.balance, "vmin": -std, "vmax": +std}
    elif "bathymetry" in legend:
        # Plotting the absolute bathymetry
        kwargs = {"cmap": cmocean.cm.deep, "vmin": 0, "vmax": 7.5}

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())
    z = ax.pcolormesh(X, Y, C, transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines()

    # Add colorbar (with correct size)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    fig.add_axes(ax_cb)
    cbar = fig.colorbar(z, cax=ax_cb)

    # Add labels
    ax.set_title(title)
    cbar.ax.set_ylabel(legend)

    # Save figure
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, title.replace(" ", "") + ".png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
