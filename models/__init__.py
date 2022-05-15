"""Collection of models for predicting bathymetry."""

import argparse

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from .physics import GDH1, H13, HS, PSM, Isostasy, Isostasy2


def get_model(args: argparse.Namespace) -> BaseEstimator:
    """Initialize a new regression model.

    Parameters:
        args: command-line arguments

    Returns:
        the model
    """
    if args.model == "linear":
        return LinearRegression(fit_intercept=False, copy_X=True, n_jobs=-1)
    elif args.model == "svr":
        return SVR(
            kernel=args.kernel,
            degree=args.degree,
            gamma=args.gamma,
            coef0=args.coef0,
            tol=args.tol,
            C=args.c,
            epsilon=args.epsilon,
            verbose=True,
        )
    elif args.model == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=[args.hidden_size] * (args.hidden_layers - 2),
            activation=args.activation,
            solver=args.solver,
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate_init=args.learning_rate,
            verbose=True,
        )
    elif args.model == "hs":
        return HS()
    elif args.model == "psm":
        return PSM()
    elif args.model == "gdh1":
        return GDH1()
    elif args.model == "h13":
        return H13()
    elif args.model == "isostasy":
        return Isostasy()
    elif args.model == "isostasy2":
        return Isostasy2()
