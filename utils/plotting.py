"""Collection of plotting utilities."""

import os

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_world(directory: str, data: np.ndarray, title: str, legend: str) -> None:
    """Plot a world map with data.

    Parameters:
        directory: the directory to save the file to
        data: the data to display
        title: the figure title
        legend: the legend label
    """
    if "difference" in legend:
        # Plotting the difference
        std = np.nanstd(data["depth"].values)
        kwargs = {"cmap": cmocean.cm.balance, "vmin": -std, "vmax": +std}
    elif "bathymetry" in legend or "depth" in legend:
        # Plotting the absolute bathymetry
        kwargs = {"cmap": cmocean.cm.deep, "vmin": 0, "vmax": 7.5}
    else:
        # Plotting thickness of sediments
        kwargs = {}

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())
    z = ax.imshow(data, transform=ccrs.PlateCarree(), **kwargs)
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
    print(f"Writing {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
