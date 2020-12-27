#!/usr/bin/env python3

"""script for hyperparameter tuning"""

import argparse
import json
import os

from utils.io import load_pickle


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('files', nargs='+', help='pickle files to compare',
                        metavar='FILE')

    return parser


def main(args: argparse.Namespace):
    """High-level pipeline.

    Compare hyperparameter performance.

    Parameters:
        args: command-line arguments
    """
    best_rmse = float('inf')
    best_r2 = 0

    best_rmse_dict: dict = {}
    best_r2_dict: dict = {}

    for filename in args.files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        data = load_pickle(os.path.dirname(filename), basename)

        rmse = data['accuracies']['validation']['RMSE']
        r2 = data['accuracies']['validation']['R^2']

        if rmse < best_rmse:
            best_rmse = rmse
            best_rmse_dict = data

        if r2 > best_r2:
            best_r2 = r2
            best_r2_dict = data

    print('\nBest validation RMSE:\n')
    print(best_rmse_dict['args'])
    print(json.dumps(best_rmse_dict['accuracies'], indent=4))

    print('\nBest validation R^2:\n')
    print(best_r2_dict['args'])
    print(json.dumps(best_r2_dict['accuracies'], indent=4))


if __name__ == '__main__':
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    main(args)
