#!/usr/bin/env python3

"""plotting tools for visualizing models."""

import argparse
from functools import partial, reduce
import json
import operator
import os

from geocube.api.core import make_geocube
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

from datasets.age import read_age
from datasets.crust import read_crust
from models.physics import GDH1, H13, HS, PSM
from preprocessing.filter import filter_crust_type, filter_nans
from preprocessing.map import spatial_join
from utils.io import load_netcdf
from utils.plotting import plot_world


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Generic arguments
    parser.add_argument(
        "-d",
        "--data-dir",
        default="data",
        help="directory containing datasets",
        metavar="DIR",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        default="checkpoints",
        help="directory to save checkpoints to",
        metavar="DIR",
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        default="results",
        help="directory to save results to",
        metavar="DIR",
    )

    # Style subparser
    subparsers = parser.add_subparsers(
        dest="style", required=True, help="style of plot to produce"
    )

    # Plot styles
    twod_parser = subparsers.add_parser("2d", help="2d cross-section")
    twod_parser.add_argument(
        "-y",
        "--year",
        default=2020,
        type=int,
        choices=[2020, 2019, 2016, 2013, 2008],
        help="year of seafloor age dataset to use",
    )

    world_parser = subparsers.add_parser("world", help="world map")
    world_parser.add_argument(
        "layers",
        nargs="+",
        choices=[
            "truth",
            "hs",
            "psm",
            "gdh1",
            "h13",
            "linear",
            "svr",
            "mlp",
            "isostasy",
            "isostasy2",
        ],
        help="layers to subtract",
    )

    layer_parser = subparsers.add_parser("layer", help="layer attributes")
    layer_parser.add_argument("layer", choices=["sediments", "moho"])

    return parser


def main_2d(args: argparse.Namespace) -> None:
    """Plot a 2d cross-section.

    Parameters:
        args: command-line arguments
    """
    print("\nReading datasets...")
    age = read_age(args.data_dir, args.year)
    crust = read_crust(args.data_dir)

    print("\nPreprocessing...")
    data = spatial_join(crust, age)
    data = filter_nans(data)
    data = filter_crust_type(data)
    X, y = data, -data["boundary topography", "upper crystalline crust"]

    print("\nPredicting...")
    x = X["age"]
    x_all_arr = np.linspace(0, np.max(x))
    x_all = pd.DataFrame({"age": x_all_arr})
    y_hs = HS().predict(x_all)
    y_psm = PSM().predict(x_all)
    y_gdh1 = GDH1().predict(x_all)
    y_h13 = H13().predict(x_all)

    print("\nPlotting...")
    plt.figure()
    plt.xlim(0, 185)
    plt.ylim(0, 14)
    plt.scatter(x, y, s=1)
    hs = plt.plot(x_all_arr, y_hs, "tab:orange")[0]
    psm = plt.plot(x_all_arr, y_psm, "tab:green")[0]
    gdh1 = plt.plot(x_all_arr, y_gdh1, "tab:red")[0]
    h13 = plt.plot(x_all_arr, y_h13, "tab:purple")[0]
    plt.title("Comparison of physical models")
    plt.xlabel("Age (Ma)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.legend([hs, psm, gdh1, h13], ["HS", "PSM", "GDH1", "H13"])

    # Save figure
    os.makedirs(args.results_dir, exist_ok=True)
    filename = os.path.join(args.results_dir, f"2d_{args.year}.png")
    print(f"Writing {filename}...")
    plt.savefig(filename, dpi=300, bbox_inches="tight")


def main_world(args: argparse.Namespace) -> None:
    """Plot a world map.

    Parameters:
        args: command-line arguments
    """
    title = " - ".join(args.layers)
    if len(args.layers) > 1:
        legend = "difference (km)"
    else:
        legend = "bathymetry (km)"

    print("\nReading layer(s)...")
    load_netcdfs = partial(load_netcdf, args.checkpoint_dir)
    layers = map(load_netcdfs, args.layers)
    data = reduce(operator.sub, layers)

    print("\nPlotting...")
    plot_world(args.results_dir, data["depth"].values, title, legend)


def main_layer(args: argparse.Namespace) -> None:
    """Plot layer attributes.

    ``args.layers`` can either be:

    * 'sediments': plot sediment thickness
    * 'moho': plot moho depth

    Parameters:
        args: command-line arguments
    """
    print("\nReading datasets...")
    crust = read_crust(args.data_dir)

    print("\nPreprocessing...")
    if args.layer == "sediments":
        bottom = crust["boundary topography", "lower sediments"]
        top = crust["boundary topography", "ice"]
        layer = bottom - top
        title = "Sediment thickness"
        legend = "thickness (km)"
    elif args.layer == "moho":
        layer = crust["boundary topography", "moho"]
        title = "Moho depth"
        legend = "depth (km)"

    df = pd.DataFrame({"layer": layer, "geometry": crust["geom"]})
    ds = make_geocube(
        vector_data=df,
        resolution=(-1, 1),
        geom=json.dumps(mapping(box(-180, -90, 180, 90))),
    )
    data = ds["layer"].to_numpy()

    print("\nPlotting...")
    plot_world(args.results_dir, data, title, legend)


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    if args.style == "2d":
        main_2d(args)
    elif args.style == "world":
        main_world(args)
    elif args.style == "layer":
        main_layer(args)
