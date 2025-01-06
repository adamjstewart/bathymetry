#!/usr/bin/env python3

"""machine learning model for predicting ocean bathymetry"""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from datasets.age import read_age
from datasets.crust import read_crust
from datasets.plate import read_plate
from metrics import evaluate, weights
from models import get_model
from preprocessing import preprocess
from preprocessing.map import inverse_standardize, standardize
from utils.io import save_checkpoint, save_prediction


def set_up_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns:
        the argument parser
    """
    # Initialize new parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Generic arguments
    parser.add_argument(
        "-d",
        "--data-dir",
        default="data",
        help="directory containing datasets",
        metavar="DIR",
    )
    parser.add_argument(
        "-c",
        "--checkpoint-dir",
        default="checkpoints",
        help="directory to save checkpoints to",
        metavar="DIR",
    )
    parser.add_argument(
        "-y",
        "--year",
        default=2020,
        type=int,
        choices=[2020, 2019, 2016, 2013, 2008],
        help="year of seafloor age dataset to use",
    )
    parser.add_argument(
        "-g",
        "--grid-size",
        default=10,
        type=int,
        help="size of grid cells (in degrees) for cross validation splitting",
    )
    parser.add_argument(
        "-a",
        "--ablation",
        help="comma-separated list of columns to drop during ablation study",
    )
    parser.add_argument(
        "-s", "--seed", default=1, type=int, help="seed for random number generation"
    )

    # Model subparser
    subparsers = parser.add_subparsers(
        dest="model", required=True, help="machine learning model"
    )

    # Machine learning models
    subparsers.add_parser("linear", help="linear regression")

    ridge_parser = subparsers.add_parser(
        "ridge",
        help="ridge regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ridge_parser.add_argument(
        "--alpha", default=100, type=float, help="L2 penalty term"
    )

    svr_parser = subparsers.add_parser(
        "svr",
        help="support vector regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    svr_parser.add_argument(
        "--kernel",
        default="rbf",
        choices=["linear", "poly", "rbf", "sigmoid", "precomputed"],
        help="kernel type",
    )
    svr_parser.add_argument(
        "--degree", default=3, type=int, help="degree of poly kernel"
    )
    svr_parser.add_argument(
        "--gamma", default="scale", help="rbf/poly/sigmoid kernel coefficient"
    )
    svr_parser.add_argument(
        "--coef0",
        default=0.0001,
        type=float,
        help="independent term in poly/sigmoid kernel",
    )
    svr_parser.add_argument(
        "--tol", default=0.001, type=float, help="tolerance for stopping criterion"
    )
    svr_parser.add_argument(
        "--c", default=10, type=float, help="regularization parameter"
    )
    svr_parser.add_argument(
        "--epsilon", default=0.1, type=float, help="epsilon in the epsilon-SVR model"
    )

    mlp_parser = subparsers.add_parser(
        "mlp",
        help="multi-layer perceptron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mlp_parser.add_argument(
        "--activation",
        default="tanh",
        choices=["identity", "logistic", "tanh", "relu"],
        help="activation function for the hidden layer",
    )
    mlp_parser.add_argument(
        "--solver",
        default="adam",
        choices=["lbfgs", "sgd", "adam"],
        help="solver for weight optimization",
    )
    mlp_parser.add_argument(
        "--alpha", default=0.1, type=float, help="L2 penalty parameter"
    )
    mlp_parser.add_argument(
        "--batch-size", default=200, type=int, help="size of minibatches"
    )
    mlp_parser.add_argument(
        "--hidden-layers", default=6, type=int, help="number of hidden layers"
    )
    mlp_parser.add_argument(
        "--hidden-size", default=256, type=int, help="size of hidden units"
    )
    mlp_parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="initial learning rate"
    )

    # Plate models
    subparsers.add_parser("hs", help="half-space cooling model")
    subparsers.add_parser("psm", help="parsons and sclater model")
    subparsers.add_parser("gdh1", help="global depth and heat flow model")
    subparsers.add_parser("h13", help="hasterok model")

    return parser


def main(args: argparse.Namespace) -> None:
    """High-level pipeline.

    Trains the model and evaluates performance.

    Args:
        args: command-line arguments
    """
    print("\nReading datasets...")
    age = read_age(args.data_dir, args.year)
    crust = read_crust(args.data_dir)
    plate = read_plate(args.data_dir)

    print("\nPreprocessing...")
    X, y, geom, groups = preprocess(age, crust, plate, args)

    print("\nCross-validation...")
    cv = GroupKFold(shuffle=True, random_state=args.seed)
    y_pred = gpd.GeoDataFrame()
    y_true = gpd.GeoDataFrame()
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Group {i}")

        # Split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        geom_train = geom.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        geom_test = geom.iloc[test_idx]

        # Standardize data
        X_train, X_test, _ = standardize(X_train, X_test, args)
        y_train, y_test, y_scaler = standardize(y_train, y_test, args)

        # Train model
        model = get_model(args)
        model.fit(X_train, y_train, sample_weight=weights(geom_train))

        # Make predictions
        y_pred_test = pd.Series(model.predict(X_test), index=y_test.index)

        # Reverse standardization
        y_test, y_pred_test = inverse_standardize(y_test, y_pred_test, y_scaler)
        y_pred_test = y_pred_test.clip(0)

        y_test = gpd.GeoDataFrame({"depth": y_test.values}, geometry=geom_test.values)
        y_pred_test = gpd.GeoDataFrame(
            {"depth": y_pred_test.values}, geometry=geom_test.values
        )
        y_true = pd.concat([y_true, y_test])
        y_pred = pd.concat([y_pred, y_pred_test])

    print("\nEvaluating...")
    accuracies = evaluate(y_true["depth"], y_pred["depth"], weights(geom))

    print("\nSaving predictions...")
    save_checkpoint(model, args, accuracies)
    save_prediction(y_true, args, "truth")
    save_prediction(y_pred, args, args.model)


if __name__ == "__main__":
    # Parse supplied arguments
    parser = set_up_parser()
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    main(args)
