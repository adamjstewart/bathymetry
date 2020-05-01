#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

from datasets.crust import read_data
from metrics import evaluate
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

    # Machine learning models
    subparsers.add_parser('linear', help='linear regression')

    # Physical models
    subparsers.add_parser('psm', help='parsons and sclater model')
    subparsers.add_parser('gdh1', help='global depth and heat flow model')

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
    X_train, X_val, X_test, y_train, y_val, y_test, y_scaler = preprocess(
        data, args)

    print('Training...')
    model = get_model(args)
    model.fit(X_train, y_train)

    print('Predicting...')
    yhat_train = model.predict(X_train)
    yhat_val = model.predict(X_val)
    yhat_test = model.predict(X_test)

    print('Postprocessing...')
    y_train, y_val, y_test = postprocess(
        y_train, y_val, y_test, y_scaler)
    yhat_train, yhat_val, yhat_test = postprocess(
        yhat_train, yhat_val, yhat_test, y_scaler)

    print('Evaluating...')
    print('\nTrain:')
    evaluate(y_train, yhat_train)
    print('\nValidation:')
    evaluate(y_val, yhat_val)
    print('\nTest:')
    evaluate(y_test, yhat_test)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
