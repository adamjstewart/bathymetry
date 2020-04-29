"""Preprocessing utilities that filter out unwanted data points.

This process is known as data cleaning.
"""

import pandas as pd


def filter_nans(data: pd.DataFrame):
    """Filter out data points containing NaN values.

    Parameters:
        data: the entire dataset
    """
    data.dropna(inplace=True)


def filter_crust_type(data: pd.DataFrame) -> pd.DataFrame:
    """Filter out continental crust and oceanic plateaus.

    The only crust types we want to keep are:
    * A0: oceans 3 Myrs and younger
    * A1: normal oceanic

    Parameters:
        data: the entire dataset
    """
    codes = ['A0', 'A1']
    return data[data['crust type', 'crust type'].isin(codes)]
