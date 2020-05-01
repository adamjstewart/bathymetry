"""Collection of models for predicting bathymetry."""

import argparse

from sklearn.linear_model import LinearRegression

from .physics import PSM, GDH1


def get_model(args: argparse.Namespace):
    """Initialize a new regression model.

    Parameters:
        args: command-line arguments

    Returns:
        the model
    """
    if args.model == 'linear':
        return LinearRegression(
            fit_intercept=False, normalize=False, copy_X=True, n_jobs=-1)
    elif args.model == 'psm':
        return PSM()
    elif args.model == 'gdh1':
        return GDH1()
