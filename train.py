#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

import numpy as np
import pandas as pd

from datasets.crust import read_data
from metrics import evaluate
from models import get_model
from preprocessing import preprocess, postprocess
from utils.io import save_pickle, save_checkpoint


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
    parser.add_argument(
        '-a', '--ablation',
        help='comma-separated list of columns to drop during ablation study')
    parser.add_argument(
        '-s', '--seed', default=1, type=int,
        help='seed for random number generation')

    # Model subparser
    subparsers = parser.add_subparsers(
        dest='model', required=True, help='machine learning model')

    # Machine learning models
    subparsers.add_parser('linear', help='linear regression')

    svr_parser = subparsers.add_parser(
        'svr', help='support vector regression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    svr_parser.add_argument(
        '--kernel', default='poly',
        choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        help='kernel type')
    svr_parser.add_argument(
        '--degree', default=3, type=int, help='degree of poly kernel')
    svr_parser.add_argument(
        '--gamma', default='auto', help='rbf/poly/sigmoid kernel coefficient')
    svr_parser.add_argument(
        '--coef0', default=0.1, type=float,
        help='independent term in poly/sigmoid kernel')
    svr_parser.add_argument(
        '--tol', default=1e-3, type=float,
        help='tolerance for stopping criterion')
    svr_parser.add_argument(
        '--c', default=4, type=float, help='regularization parameter')
    svr_parser.add_argument(
        '--epsilon', default=0.1, type=float,
        help='epsilon in the epsilon-SVR model')

    mlp_parser = subparsers.add_parser(
        'mlp', help='multi-layer perceptron',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mlp_parser.add_argument(
        '--activation', default='relu',
        choices=['identity', 'logistic', 'tanh', 'relu'],
        help='activation function for the hidden layer')
    mlp_parser.add_argument(
        '--solver', default='adam', choices=['lbfgs', 'sgd', 'adam'],
        help='solver for weight optimization')
    mlp_parser.add_argument(
        '--alpha', default=0.0001, type=float, help='L2 penalty parameter')
    mlp_parser.add_argument(
        '--batch-size', default=200, type=int, help='size of minibatches')
    mlp_parser.add_argument(
        '--hidden-layers', default=7, type=int, help='number of hidden layers')
    mlp_parser.add_argument(
        '--hidden-size', default=512, type=int, help='size of hidden units')
    mlp_parser.add_argument(
        '--learning-rate', default=0.0001, type=float,
        help='initial learning rate')

    # Physical models
    subparsers.add_parser('hs', help='half-space cooling model')
    subparsers.add_parser('psm', help='parsons and sclater model')
    subparsers.add_parser('gdh1', help='global depth and heat flow model')
    subparsers.add_parser('h13', help='hasterok model')
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
    accuracies = {}
    print('\nTrain:')
    accuracies['train'] = evaluate(y_train, yhat_train)
    print('\nValidation:')
    accuracies['validation'] = evaluate(y_val, yhat_val)
    print('\nTest:')
    accuracies['test'] = evaluate(y_test, yhat_test)
    print()

    print('Saving...')
    y = pd.concat([y_train, y_val, y_test])
    yhat = pd.concat([yhat_train, yhat_val, yhat_test])
    save_pickle(y, args.checkpoint_dir, 'truth')
    save_pickle(yhat, args.checkpoint_dir, args.model)
    save_checkpoint(model, args, accuracies)


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    main(args)
