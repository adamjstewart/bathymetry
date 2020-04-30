#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

from datasets.crust import read_data
from preprocessing import preprocess


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

    # Model subparser
    subparsers = parser.add_subparsers(
        dest='model', required=True, help='machine learning model')

    lstsq_parser = subparsers.add_parser('lstsq', help='least squares')

    return parser


def main(args: argparse.Namespace):
    """High-level pipeline.

    Trains the model and evaluates performance.

    Parameters:
        args: command-line arguments
    """
    data = read_data(args.data_dir)

    X, y = preprocess(data)

    print(X)
    print(y)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
