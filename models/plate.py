"""A collection of plate models.

https://scikit-learn.org/stable/developers/develop.html
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PlateModel(BaseEstimator, RegressorMixin):
    """Base class for all plate models."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.typing.NDArray[np.float64]:
        """Sediment correction factor.

        Args:
            X: the dataset

        Returns:
            the prediction
        """
        rho_us = X["density", "upper sediments"].values
        rho_ms = X["density", "middle sediments"].values
        rho_ls = X["density", "lower sediments"].values
        rho_w = X["density", "water"].values
        rho_m = X["density", "moho"].values
        h_us = X["thickness", "upper sediments"].values
        h_ms = X["thickness", "middle sediments"].values
        h_ls = X["thickness", "lower sediments"].values
        h_s = h_us + h_ms + h_ls

        factor: np.typing.NDArray[np.float64] = (
            rho_us * h_us + rho_ms * h_ms + rho_ls * h_ls - rho_w * h_s
        ) / (rho_m - rho_w)
        return factor


class HS(PlateModel):
    """Half-Space cooling model (HS).

    Fundamentals of Ridge Crest Topography.

    Davis & Lister, Earth and Planetary Science Letters, 1974

    https://doi.org/10.1016/0012-821X(74)90180-0
    http://osu-wams-blogs-uploads.s3.amazonaws.com/blogs.dir/2281/files/2015/08/DavisLister_EPSL74.pdf
    """

    def predict(self, X: pd.DataFrame) -> np.typing.NDArray[np.float64]:
        """Predict bathymetry based on age.

        Args:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values * 3.154e13
        rho_0 = 3300
        rho_w = 1000
        alpha = 4e-5
        kappa = 8e-7
        T_1 = 1220

        depth: np.typing.NDArray[np.float64] = 2.5 + (
            (2 * rho_0 * alpha * T_1)
            / (rho_0 - rho_w)
            * np.sqrt(kappa * t / np.pi)
            / 1000
        )

        return depth


class PSM(PlateModel):
    """Parsons and Sclater Model (PSM).

    An analysis of the variation of ocean floor bathymetry and
    heat flow with age.

    Parsons & Sclater, Journal of geophysical research, 1977

    https://doi.org/10.1029/JB082i005p00803
    https://pdfs.semanticscholar.org/a67e/9d46e6b1cd7a956e8e5976d87b64b5f1f7df.pdf
    """

    def predict(self, X: pd.DataFrame) -> np.typing.NDArray[np.float64]:
        """Predict bathymetry based on age.

        Args:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        depth: np.typing.NDArray[np.float64] = np.where(
            t < 70,
            # Young crust
            2.5 + 0.35 * t**0.5,
            # Old crust
            6.4 - 3.2 * np.exp(-t / 62.8),
        )

        return depth + super().predict(X)


class GDH1(PlateModel):
    """Global Depth and Heat Flow (GDH1) model.

    A model for the global variation in oceanic depth and
    heat flow with lithospheric age.

    Stein & Stein, Nature, 1992

    https://doi.org/10.1038/359123a0
    https://physics.unm.edu/Courses/Roy/Phys480_581Fa14/papers/Stein_Stein_359123a0.pdf
    """

    def predict(self, X: pd.DataFrame) -> np.typing.NDArray[np.float64]:
        """Predict bathymetry based on age.

        Args:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        depth: np.typing.NDArray[np.float64] = np.where(
            t < 20,
            # Young crust
            2.6 + 0.365 * t**0.5,
            # Old crust
            5.651 - 2.473 * np.exp(-0.0278 * t),
        )

        return depth + super().predict(X)


class H13(PlateModel):
    """Hasterok model.

    A heat flow based cooling model for tectonic plates.

    Hasterok, Earth and Planetary Science Letters, 2013

    https://doi.org/10.1016/j.epsl.2012.10.036
    https://www.academia.edu/download/50241193/Hasterok2013.pdf
    """

    def predict(self, X: pd.DataFrame) -> np.typing.NDArray[np.float64]:
        """Predict bathymetry based on age.

        Args:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        depth: np.typing.NDArray[np.float64] = np.where(
            t <= 17.4,
            # Young crust
            0.4145 * t**0.5,
            # Old crust
            3.109 - 2.520 * np.exp(-0.034607 * t),
        )

        # Ridge depth is average of 2.424 and 2.514 km
        depth += 2.469

        return depth + super().predict(X)
