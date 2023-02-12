#!/usr/bin/env python3

"""plotting tools for visualizing models."""

import argparse
import json
import operator
import os
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geocube.api.core import make_geocube
from shapely.geometry import box, mapping

from datasets.age import read_age
from datasets.crust import read_crust
from models.plate import GDH1, H13, HS, PSM
from preprocessing.filter import filter_crust_type, filter_nans
from preprocessing.map import boundary_to_thickness, spatial_join
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
    parser.add_argument(
        "-y",
        "--year",
        default=2020,
        type=int,
        choices=[2020, 2019, 2016, 2013, 2008],
        help="year of seafloor age dataset to use",
    )

    # Style subparser
    subparsers = parser.add_subparsers(
        dest="style", required=True, help="style of plot to produce"
    )

    # Plot styles
    subparsers.add_parser("2d", help="2d cross-section")

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
            "ridge",
            "svr",
            "mlp",
        ],
        help="layers to subtract",
    )

    feature_parser = subparsers.add_parser("feature", help="features")
    feature_parser.add_argument(
        "layer",
        choices=["water", "ice", "sediments", "crust", "moho", "age"],
        help="layer to plot",
    )
    feature_parser.add_argument(
        "feature",
        choices=["thickness", "p", "s", "density", "age"],
        help="feature to plot",
    )

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
    plt.title("Comparison of plate models")
    plt.xlabel("Age (Ma)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.legend([hs, psm, gdh1, h13], ["HS", "PSM", "GDH1", "H13"])

    # Save figure
    directory = os.path.join(args.results_dir, "plate")
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"2d_{args.year}.png")
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
        directory = os.path.join(args.results_dir, "residual")
    else:
        legend = "bathymetry (km)"
        directory = os.path.join(args.results_dir, "bathymetry")

    print("\nReading layer(s)...")
    load_netcdfs = partial(load_netcdf, args.checkpoint_dir)
    layers = map(load_netcdfs, args.layers)
    data = reduce(operator.sub, layers)

    print("\nPlotting...")
    plot_world(directory, data["depth"].values, title, legend)


def main_feature(args: argparse.Namespace) -> None:
    """Plot features.

    Parameters:
        args: command-line arguments
    """
    print("\nReading datasets...")
    age = read_age(args.data_dir, args.year)
    crust = read_crust(args.data_dir)

    print("\nPreprocessing...")
    df = spatial_join(crust, age)
    df = boundary_to_thickness(df)

    # Feature
    feature = args.feature
    feature_map = {
        "p": "p-wave velocity",
        "s": "s-wave velocity",
    }
    if args.feature in feature_map:
        feature = feature_map[args.feature]

    # Layer
    layer = args.layer
    if layer in ["water", "ice", "moho"]:
        s = df[feature, layer]
    elif layer == "sediments":
        s = (
            df[feature, "upper sediments"]
            + df[feature, "middle sediments"]
            + df[feature, "lower sediments"]
        )
        if feature != "thickness":
            s /= 3
    elif layer == "crust":
        s = (
            df[feature, "upper crystalline crust"]
            + df[feature, "middle crystalline crust"]
            + df[feature, "lower crystalline crust"]
        )
        if feature != "thickness":
            s /= 3
    elif layer == "age":
        s = df[feature, ""]

    # Vector to matrix
    df = pd.DataFrame({"feature": s, "geometry": df["geom"]})
    ds = make_geocube(
        vector_data=df,
        resolution=(-1, 1),
        geom=json.dumps(mapping(box(-180, -90, 180, 90))),
    )
    data = ds["feature"].to_numpy()

    # Title
    title = "_".join([layer, feature])

    # Legend
    legend_map = {
        "thickness": "thickness (km)",
        "p": "velocity (km/s)",
        "s": "velocity (km/s)",
        "density": "density (g/cm$^3$)",
        "age": "age (Ma)",
    }
    legend = legend_map[args.feature]

    print("\nPlotting...")
    directory = os.path.join(args.results_dir, "features")
    plot_world(directory, data, title, legend)


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    if args.style == "2d":
        main_2d(args)
    elif args.style == "world":
        main_world(args)
    elif args.style == "feature":
        main_feature(args)
