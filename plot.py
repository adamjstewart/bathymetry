#!/usr/bin/env python3

"""plotting tools for visualizing models."""

import argparse
from functools import partial, reduce
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.crust import read_crust
from models.physics import GDH1, H13, HS, PSM
from preprocessing.filter import filter_crust_type, filter_nans
from utils.io import load_pickle
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
            "svr",
            "mlp",
            "isostasy",
            "isostasy2",
        ],
        help="layers to subtract",
    )

    return parser


def main_2d(args: argparse.Namespace) -> None:
    """Plot a 2d cross-section.

    Parameters:
        args: command-line arguments
    """
    print("Reading dataset...")
    data = read_crust(args.data_dir)

    print("Preprocessing...")
    data = filter_nans(data)
    data = filter_crust_type(data)
    X, y = data, -data["boundary topograpy", "upper crystalline crust"]

    print("Predicting...")
    x = X["age", "age"]
    x_all = np.linspace(0, np.max(x))
    x_all = pd.DataFrame(x_all, columns=pd.MultiIndex.from_product([["age"], ["age"]]))
    y_hs = HS().predict(x_all)
    y_psm = PSM().predict(x_all)
    y_gdh1 = GDH1().predict(x_all)
    y_h13 = H13().predict(x_all)

    print("Plotting...")
    plt.figure()
    plt.xlim(0, 185)
    plt.ylim(0, 14)
    plt.scatter(x, y, s=1)
    hs = plt.plot(x_all, y_hs, "tab:orange")[0]
    psm = plt.plot(x_all, y_psm, "tab:green")[0]
    gdh1 = plt.plot(x_all, y_gdh1, "tab:red")[0]
    h13 = plt.plot(x_all, y_h13, "tab:purple")[0]
    plt.title("Comparison of physical models")
    plt.xlabel("Age (Ma)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.legend([hs, psm, gdh1, h13], ["HS", "PSM", "GDH1", "H13"])

    # Save figure
    os.makedirs(args.results_dir, exist_ok=True)
    filename = os.path.join(args.results_dir, "2d.png")
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

    print("Reading layer(s)...")
    load_pickles = partial(load_pickle, args.checkpoint_dir)
    layers = map(load_pickles, args.layers)

    print("Processing...")
    data = reduce(operator.sub, layers)

    print("Plotting...")
    plot_world(args.results_dir, data, title, legend)


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    if args.style == "2d":
        main_2d(args)
    elif args.style == "world":
        main_world(args)
