"""A collection of physical models."""

import numpy as np
import pandas as pd


class PSM:
    """Parsons and Sclater Model (PSM).

    An analysis of the variation of ocean floor bathymetry and
    heat flow with age.

    Parsons & Sclater, Journal of geophysical research, 1977

    https://doi.org/10.1029/JB082i005p00803
    https://pdfs.semanticscholar.org/a67e/9d46e6b1cd7a956e8e5976d87b64b5f1f7df.pdf
    """
    def fit(self, X, y):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on age.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        t = X['age', 'age'].values

        return np.where(
            t < 70,
            # Young crust
            2.5 + 0.35 * t**0.5,
            # Old crust
            6.4 - 3.2 * np.exp(-t / 62.8)
        )


class GDH1:
    """Global Depth and Heat Flow (GDH1) model.

    A model for the global variation in oceanic depth and
    heat flow with lithospheric age.

    Stein & Stein, Nature, 1992

    https://doi.org/10.1038/359123a0
    http://physics.unm.edu/Courses/Roy/Phys480_581Fa14/papers/Stein_Stein_359123a0.pdf
    """
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
