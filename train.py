#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

from datasets.crust import read_data
from models import get_model
from preprocessing import preprocess, postprocess


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

    subparsers.add_parser('linear', help='linear regression')

    return parser


def main(args: argparse.Namespace):
    """High-level pipeline.

    Trains the model and evaluates performance.

    Parameters:
        args: command-line arguments
    """
    print('Reading dataset...')
    data = read_data(args.data_dir)

    print('Preprocessing...')
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = preprocess(data)

    print('Training...')
    model = get_model(args)
    model.fit(X_train, y_train)

    print('Predicting...')
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    print('Postprocessing...')
    yhat_train, yhat_val, yhat_test = postprocess(
        yhat_train, yhat_val, yhat_test, y_scaler)

    print('Evaluating...')


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
