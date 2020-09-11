#!/usr/bin/env python3

"""plotting tools for visualizing models."""

import argparse
import operator
from functools import partial, reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.crust import read_data
from models.physics import PSM, GDH1
from preprocessing.filter import filter_nans, filter_crust_type
from utils.io import load_pickle
from utils.plotting import plot_world


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Generic arguments
    parser.add_argument(
        '-d', '--data-dir', default='data/CRUST1.0',
        help='directory containing CRUST1.0 dataset', metavar='DIR')
    parser.add_argument(
        '-c', '--checkpoint-dir', default='checkpoints',
        help='directory to save checkpoints to', metavar='DIR')

    # Style subparser
    subparsers = parser.add_subparsers(
        dest='style', required=True, help='style of plot to produce')

    # Plot styles
    subparsers.add_parser('2d', help='2d cross-section')

    world_parser = subparsers.add_parser('world', help='world map')
    world_parser.add_argument(
        'layers', nargs='+', choices=[
            'truth', 'psm', 'gdh1', 'linear', 'svr', 'isostasy', 'isostasy2'],
        help='layers to subtract')

    return parser


def main_2d(args: argparse.Namespace):
    """Plot a 2d cross-section.

    Parameters:
        args: command-line arguments
    """
    print('Reading dataset...')
    data = read_data(args.data_dir)

    print('Preprocessing...')
    data = filter_nans(data)
    data = filter_crust_type(data)
    X, y = data, -data['boundary topograpy', 'upper crystalline crust']

    print('Predicting...')
    x = X['age', 'age']
    x_psm_gdh1 = np.linspace(0, np.max(x))
    x_psm_gdh1 = pd.DataFrame(
        x_psm_gdh1, columns=pd.MultiIndex.from_product([['age'], ['age']]))
    y_psm = PSM().predict(x_psm_gdh1)
    y_gdh1 = GDH1().predict(x_psm_gdh1)
    x_psm_gdh1 = x_psm_gdh1.values

    print('Plotting...')
    plt.figure()
    plt.scatter(x, y, s=1)
    psm, = plt.plot(x_psm_gdh1, y_psm, 'm-')
    gdh1, = plt.plot(x_psm_gdh1, y_gdh1, 'y-')
    plt.title('Comparison of PSM and GDH1 models')
    plt.xlabel('Age (Ma)')
    plt.ylabel('Depth (km)')
    plt.gca().invert_yaxis()
    plt.legend([psm, gdh1], ['PSM', 'GDH1'])
    plt.show()


def main_world(args: argparse.Namespace):
    """Plot a world map.

    Parameters:
        args: command-line arguments
    """
    title = ' - '.join(args.layers)
    if len(args.layers) > 1:
        legend = 'difference (km)'
    else:
        legend = 'bathymetry (km)'

    print('Reading layer(s)...')
    load_pickles = partial(load_pickle, args.checkpoint_dir)
    layers = map(load_pickles, args.layers)

    print('Processing...')
    data = reduce(operator.sub, layers)

    print('Plotting...')
    plot_world(data, title, legend)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    if args.style == '2d':
        main_2d(args)
    elif args.style == 'world':
        main_world(args)
