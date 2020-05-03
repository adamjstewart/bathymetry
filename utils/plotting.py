"""Collection of plotting utilities."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def plot_world(data: pd.DataFrame):
    """Plot a world map with data.

    Parameters:
        data: the data to display
    """
    ax = plt.axes(projection=ccrs.Mollweide())
    ax.stock_img()
    ax.coastlines()
    plt.show()
