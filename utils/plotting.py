"""Collection of plotting utilities."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd


def plot_world(data: pd.Series):
    """Plot a world map with data.

    Parameters:
        data: the data to display
    """
    assert isinstance(data, pd.Series)

    # Convert from MultiIndex Series to Index DataFrame
    data = data.reset_index()
    lat = data['latitude']
    lon = data['longitude']
    z = data[0]

    ax = plt.axes(projection=ccrs.Mollweide())
    ax.stock_img()
    ax.coastlines()
    plt.contourf(lon, lat, z, 60, transform=ccrs.Mollweide())
    plt.show()
