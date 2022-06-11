"""Collection of plotting utilities."""

import os

import cartopy.crs as ccrs
import cmocean
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_world(
    directory: str, data: np.typing.NDArray[np.float_], title: str, legend: str
) -> None:
    """Plot a world map with data.

    Parameters:
        directory: the directory to save the file to
        data: the data to display
        title: the figure title
        legend: the legend label
    """
    kwargs = {}

    # Colormap
    cmap_map = {
        "difference": cmocean.cm.balance,
        "bathymetry": cmocean.cm.deep,
        "velocity": cmocean.cm.speed,
        "density": cmocean.cm.dense,
        "age": plt.get_cmap("gist_rainbow"),
    }
    for key, value in cmap_map.items():
        if key in legend:
            kwargs.update({"cmap": value})
            break

    # Min/max values
    if "difference" in legend:
        std = np.nanstd(data)
        kwargs.update({"vmin": -std, "vmax": +std})
    elif "thickness" in legend or "velocity" in legend or "density" in legend:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        kwargs.update({"vmin": mean - std, "vmax": mean + std})
    elif "bathymetry" in legend:
        kwargs.update({"vmin": 0, "vmax": 7.5})

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
