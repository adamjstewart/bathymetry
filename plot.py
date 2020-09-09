#!/usr/bin/env python3

"""plotting tools for visualizing physical models."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.crust import read_data
from models.physics import PSM, GDH1
from preprocessing.filter import filter_nans, filter_crust_type


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

    return parser


def main(args: argparse.Namespace):
    """High-level pipeline.

    Plots the performance of PSM and GDH1 models.

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


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
