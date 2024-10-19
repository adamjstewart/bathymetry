"""Collection of plotting utilities."""

import os

import cartopy.crs as ccrs
import cmocean
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_world(
    directory: str,
    data: np.typing.NDArray[np.float64],
    title: str,
    legend: str,
    plate: gpd.GeoDataFrame,
) -> None:
    """Plot a world map with data.

    Args:
        directory: the directory to save the file to
        data: the data to display
        title: the figure title
        legend: the legend label
        plate: plate boundaries
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
        kwargs.update({"vmin": -2 * std, "vmax": +2 * std})
    elif "thickness" in legend or "velocity" in legend or "density" in legend:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        kwargs.update({"vmin": mean - 2 * std, "vmax": mean + 2 * std})
    elif "bathymetry" in legend:
        kwargs.update({"vmin": 0, "vmax": 7.5})

    # Plotting
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.Mollweide())
    z = ax.imshow(data, transform=ccrs.PlateCarree(), **kwargs)
    ax.coastlines()  # type: ignore[attr-defined]
    ax = plate.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )

    # Add colorbar (with correct size)
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=Axes)
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
