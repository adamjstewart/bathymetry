#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

from datasets.crust import read_data


def set_up_parser():
    """Set up the argument parser.

    Returns:
        argparse.ArgumentParser: the argument parser
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


def main(args):
    """High-level pipeline.

    Trains the model and evaluates performance.

    Parameters:
        args (argparse.Namespace): command-line arguments
    """
    X, y = read_data(args.data_dir)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
