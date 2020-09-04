#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

import pandas as pd

from datasets.crust import read_data
from metrics import evaluate
from models import get_model
from preprocessing import preprocess, postprocess
#from utils.plotting import plot_world


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

    svr_parser = subparsers.add_parser('svr', help='support vector regression')
    svr_parser.add_argument(
        '--kernel', default='rbf',
        choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        help='kernel type')
    svr_parser.add_argument(
        '--degree', default=3, type=int, help='degree of poly kernel')
    svr_parser.add_argument(
        '--gamma', default='scale', help='rbf/poly/sigmoid kernel coefficient')
    svr_parser.add_argument(
        '--coef0', default=0, type=float,
        help='independent term in poly/sigmoid kernel')
    svr_parser.add_argument(
        '--tol', default=1e-3, type=float,
        help='tolerance for stopping criterion')
    svr_parser.add_argument(
        '--c', default=1, type=float, help='regularization parameter')
    svr_parser.add_argument(
        '--epsilon', default=0.1, type=float,
        help='epsilon in the epsilon-SVR model')

    # Physical models
    subparsers.add_parser('psm', help='parsons and sclater model')
    subparsers.add_parser('gdh1', help='global depth and heat flow model')
    subparsers.add_parser('isostasy', help='pure isostasy prediction')
    subparsers.add_parser('isostasy2', help='pure isostasy prediction')

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

    #with pd.option_context('display.max_columns', 999):
    #    print(X_train.columns)
    #    print(X_train.describe(include='all'))

    print('Training...')
    model = get_model(args)
    model.fit(X_train, y_train)

    print('Predicting...')
    yhat_train = model.predict(X_train)
    yhat_train = pd.Series(yhat_train, index=y_train.index)
    yhat_val = model.predict(X_val)
    yhat_val = pd.Series(yhat_val, index=y_val.index)
    yhat_test = model.predict(X_test)
    yhat_test = pd.Series(yhat_test, index=y_test.index)

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
    print()

    #print('Plotting...')
    #y = pd.concat([y_train, y_val, y_test])
    #yhat = pd.concat([yhat_train, yhat_val, yhat_test])
    #plot_world(y - yhat)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
