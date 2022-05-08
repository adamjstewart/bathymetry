"""A collection of physical models.

https://scikit-learn.org/stable/developers/develop.html
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.base import BaseEstimator, RegressorMixin


class HS(BaseEstimator, RegressorMixin):
    """Half-Space cooling model (HS).

    Fundamentals of Ridge Crest Topography.

    Davis & Lister, Earth and Planetary Science Letters, 1974

    https://doi.org/10.1016/0012-821X(74)90180-0
    http://osu-wams-blogs-uploads.s3.amazonaws.com/blogs.dir/2281/files/2015/08/DavisLister_EPSL74.pdf
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on age.

        Parameters:
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

        depth: np.ndarray = 2.5 + (
            (2 * rho_0 * alpha * T_1)
            / (rho_0 - rho_w)
            * np.sqrt(kappa * t / np.pi)
            / 1000
        )

        return depth


class PSM(BaseEstimator, RegressorMixin):
    """Parsons and Sclater Model (PSM).

    An analysis of the variation of ocean floor bathymetry and
    heat flow with age.

    Parsons & Sclater, Journal of geophysical research, 1977

    https://doi.org/10.1029/JB082i005p00803
    https://pdfs.semanticscholar.org/a67e/9d46e6b1cd7a956e8e5976d87b64b5f1f7df.pdf
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on age.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        depth: np.ndarray = np.where(
            t < 70,
            # Young crust
            2.5 + 0.35 * t**0.5,
            # Old crust
            6.4 - 3.2 * np.exp(-t / 62.8),
        )

        return depth


class GDH1(BaseEstimator, RegressorMixin):
    """Global Depth and Heat Flow (GDH1) model.

    A model for the global variation in oceanic depth and
    heat flow with lithospheric age.

    Stein & Stein, Nature, 1992

    https://doi.org/10.1038/359123a0
    https://physics.unm.edu/Courses/Roy/Phys480_581Fa14/papers/Stein_Stein_359123a0.pdf
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on age.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        depth: np.ndarray = np.where(
            t < 20,
            # Young crust
            2.6 + 0.365 * t**0.5,
            # Old crust
            5.651 - 2.473 * np.exp(-0.0278 * t),
        )

        return depth


class H13(BaseEstimator, RegressorMixin):
    """Hasterok model.

    A heat flow based cooling model for tectonic plates.

    Hasterok, Earth and Planetary Science Letters, 2013

    https://doi.org/10.1016/j.epsl.2012.10.036
    https://www.academia.edu/download/50241193/Hasterok2013.pdf
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on age.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        t = X["age"].values

        # Ridge depth is average of 2.424 and 2.514 km
        depth: np.ndarray = 2.469 + np.where(
            t <= 17.4,
            # Young crust
            0.4145 * t**0.5,
            # Old crust
            3.109 - 2.520 * np.exp(-0.034607 * t),
        )

        return depth


class Isostasy(BaseEstimator, RegressorMixin):
    """Simple model based on isostasy."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """Record average values of thickness and density.

        Parameters:
            X: the dataset
            y: the depths
        """
        # Thicknesses
        self.i = X["thickness", "ice"].mean()

        self.s1 = X["thickness", "upper sediments"].mean()
        self.s2 = X["thickness", "middle sediments"].mean()
        self.s3 = X["thickness", "lower sediments"].mean()

        self.c1 = X["thickness", "upper crystalline crust"].mean()
        self.c2 = X["thickness", "middle crystalline crust"].mean()
        self.c3 = X["thickness", "lower crystalline crust"].mean()

        self.w = y.mean() - self.s1 - self.s2 - self.s3

        # Densities
        self.rho_i = X["density", "ice"]
        self.rho_i = np.ma.masked_values(self.rho_i, 0).mean()

        self.rho_w = X["density", "water"]
        self.rho_w = np.ma.masked_values(self.rho_w, 0).mean()

        self.rho_s1 = X["density", "upper sediments"]
        self.rho_s2 = X["density", "middle sediments"]
        self.rho_s3 = X["density", "lower sediments"]
        self.rho_s1 = np.ma.masked_values(self.rho_s1, 0).mean()
        self.rho_s2 = np.ma.masked_values(self.rho_s2, 0).mean()
        self.rho_s3 = np.ma.masked_values(self.rho_s3, 0).mean()

        self.rho_c1 = X["density", "upper crystalline crust"]
        self.rho_c2 = X["density", "middle crystalline crust"]
        self.rho_c3 = X["density", "lower crystalline crust"]
        self.rho_c1 = np.ma.masked_values(self.rho_c1, 0).mean()
        self.rho_c2 = np.ma.masked_values(self.rho_c2, 0).mean()
        self.rho_c3 = np.ma.masked_values(self.rho_c3, 0).mean()

        self.rho_m = X["density", "moho"]
        self.rho_m = np.ma.masked_values(self.rho_m, 0).mean()

        # Calculations
        self.numerator = (
            (self.rho_i * self.i)
            + self.w * (self.rho_w - self.rho_m)
            + self.s1 * (self.rho_s1 - self.rho_m)
            + self.s2 * (self.rho_s2 - self.rho_m)
            + self.s3 * (self.rho_s3 - self.rho_m)
            + self.c1 * (self.rho_c1 - self.rho_m)
            + self.c2 * (self.rho_c2 - self.rho_m)
            + self.c3 * (self.rho_c3 - self.rho_m)
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on isostasy.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        # Thicknesses
        i = X["thickness", "ice"].values

        s1 = X["thickness", "upper sediments"].values
        s2 = X["thickness", "middle sediments"].values
        s3 = X["thickness", "lower sediments"].values

        c1 = X["thickness", "upper crystalline crust"].values
        c2 = X["thickness", "middle crystalline crust"].values
        c3 = X["thickness", "lower crystalline crust"].values

        # Densities
        rho_i = X["density", "ice"].values

        rho_w = X["density", "water"].values

        rho_s1 = X["density", "upper sediments"].values
        rho_s2 = X["density", "middle sediments"].values
        rho_s3 = X["density", "lower sediments"].values

        rho_c1 = X["density", "upper crystalline crust"].values
        rho_c2 = X["density", "middle crystalline crust"].values
        rho_c3 = X["density", "lower crystalline crust"].values

        # Pre-condition: assume that mantle density is a constant,
        # otherwise the problem is underconstrained
        rho_m = self.rho_m

        # Calculations
        numerator = (
            self.numerator
            + (rho_i * i)
            + s1 * (rho_s1 - rho_m)
            + s2 * (rho_s2 - rho_m)
            + s3 * (rho_s3 - rho_m)
            + c1 * (rho_c1 - rho_m)
            + c2 * (rho_c2 - rho_m)
            + c3 * (rho_c3 - rho_m)
        )
        denominator = rho_w - rho_m

        w = numerator / denominator

        depth: np.ndarray = w + s1 + s2 + s3

        return depth


class Isostasy2(BaseEstimator, RegressorMixin):
    """Simple model based on isostasy."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        """Record average values of thickness and density.

        Parameters:
            X: the dataset
            y: the depths
        """
        p0 = np.array([0.0, 0.0])
        self.popt, _ = opt.curve_fit(isostasy, X, y, p0)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict bathymetry based on isostasy.

        Parameters:
            X: the dataset

        Returns:
            the prediction
        """
        return isostasy(X, *self.popt)


def isostasy(X: pd.DataFrame, a: float, b: float) -> np.ndarray:
    """Predict y based on X, a, and b.

    Parameters:
        X: the dataset
        a: the total mass constant
        b: the total thickness constant

    Returns:
        y: predictions
    """
    # Thicknesses
    i = X["thickness", "ice"].values

    s1 = X["thickness", "upper sediments"].values
    s2 = X["thickness", "middle sediments"].values
    s3 = X["thickness", "lower sediments"].values

    c1 = X["thickness", "upper crystalline crust"].values
    c2 = X["thickness", "middle crystalline crust"].values
    c3 = X["thickness", "lower crystalline crust"].values

    # Densities
    rho_i = X["density", "ice"].values

    rho_w = X["density", "water"].values

    rho_s1 = X["density", "upper sediments"].values
    rho_s2 = X["density", "middle sediments"].values
    rho_s3 = X["density", "lower sediments"].values

    rho_c1 = X["density", "upper crystalline crust"].values
    rho_c2 = X["density", "middle crystalline crust"].values
    rho_c3 = X["density", "lower crystalline crust"].values

    rho_m = X["density", "moho"].values

    # Calculations
    depth: np.ndarray = (
        a
        + rho_m * c1
        + rho_m * c2
        + rho_m * c3
        - rho_i * i
        - rho_w * s1
        - rho_w * s2
        - rho_w * s3
        - rho_s1 * s1
        - rho_s2 * s2
        - rho_s3 * s3
        - rho_c1 * c1
        - rho_c2 * c2
        - rho_c3 * c3
        - rho_m * b
    ) / (rho_w - rho_m)

    return depth
