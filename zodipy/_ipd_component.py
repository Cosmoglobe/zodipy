"""

Implementation of the Interplanetary Dust components in the DIRBE model.

Custom Interplanetary Dust components may be implemented. They would need
to inherit from `Component` and implement the `compute_density` method.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ComponentLabel(Enum):
    """Labels representing the components in the DIRBE model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"


@dataclass
class Component(ABC):
    """Base class for an Interplanetary Dust Component.

    Parameters
    ----------
    x_0
        x-offset from the Sun in heliocentric ecliptic coordinates [AU].
    y_0
        y-offset from the Sun in heliocentric ecliptic coordinates [AU].
    z_0
        z-offset from the Sun in heliocentric ecliptic coordinates [AU].
    i
        Inclination with respect to the ecliptic plane [deg].
    Omega
        Ascending node [deg].
    """

    x_0: float
    y_0: float
    z_0: float
    i: float
    Omega: float

    def __post_init__(self) -> None:
        # Offset vector with broadcastable shape (3, 1, 1)
        self.X_0 = np.array([self.x_0, self.y_0, self.z_0]).reshape(3, 1, 1)

        # Frequently used quantities
        self.sin_i_rad = np.sin(np.radians(self.i))
        self.cos_i_rad = np.cos(np.radians(self.i))
        self.sin_Omega_rad = np.sin(np.radians(self.Omega))
        self.cos_Omega_rad = np.cos(np.radians(self.Omega))

    @abstractmethod
    def compute_density(
        self,
        X_helio: NDArray[np.floating],
        *,
        X_earth: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Returns the dust density of a component at points in the Solar System
        given by 'X_helio'.

        Parameters
        ----------
        X_helio
            Heliocentric ecliptic coordinates (x, y, z) of points in the Solar
            System [AU].
        X_earth
            Heliocentric ecliptic coordinates of the Earth. Required for the
            Earth-trailing Feature [AU].

        Returns
        -------
            Density of the component at the points given by 'X_helio' [1/AU].
        """


@dataclass
class Cloud(Component):
    """DIRBE Diffuse Cloud.

    Parameters
    ----------
    n_0
        Density at 1 AU.
    alpha
        Radial power-law exponent.
    beta
       Vertical shape parameter.
    gamma
        Vertical power-law exponent.
    mu
        Widening parameter for the modified fan.
    """

    n_0: float
    alpha: float
    beta: float
    gamma: float
    mu: float

    def compute_density(
        self, X_helio: NDArray[np.floating], **_
    ) -> NDArray[np.floating]:
        """See base class for documentation."""

        X_cloud = X_helio - self.X_0
        R_cloud = np.sqrt(X_cloud[0] ** 2 + X_cloud[1] ** 2 + X_cloud[2] ** 2)

        Z_cloud = (
            X_cloud[0] * self.sin_Omega_rad * self.sin_i_rad
            - X_cloud[1] * self.cos_Omega_rad * self.sin_i_rad
            + X_cloud[2] * self.cos_i_rad
        )

        ζ = np.abs(Z_cloud / R_cloud)
        μ = self.mu
        g = np.zeros_like(ζ)

        condition = ζ < μ
        g[condition] = ζ[condition] ** 2 / (2 * μ)
        g[~condition] = ζ[~condition] - (μ / 2)

        return self.n_0 * R_cloud**-self.alpha * np.exp(-self.beta * g**self.gamma)


@dataclass
class Band(Component):
    """DIRBe Dust Band.

    Parameters
    ----------
    n_0
        Density at 3 AU.
    delta_zeta
        Shape parameter [deg].
    v
        Shape parameter.
    p
        Shape parameter.
    delta_r
        Inner radial cutoff.
    """

    n_0: float
    delta_zeta: float
    v: float
    p: float
    delta_r: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.delta_zeta_rad = np.radians(self.delta_zeta)

    def compute_density(
        self, X_helio: NDArray[np.floating], **_
    ) -> NDArray[np.floating]:
        """See base class for documentation."""

        X_band = X_helio - self.X_0
        R_band = np.sqrt(X_band[0] ** 2 + X_band[1] ** 2 + X_band[2] ** 2)

        Z_band = (
            X_band[0] * self.sin_Omega_rad * self.sin_i_rad
            - X_band[1] * self.cos_Omega_rad * self.sin_i_rad
            + X_band[2] * self.cos_i_rad
        )

        ζ = np.abs(Z_band / R_band)
        ζ_over_δ_ζ = ζ / self.delta_zeta_rad
        term1 = 3 * self.n_0 / R_band
        term2 = np.exp(-(ζ_over_δ_ζ**6))

        # Differs from eq 8 in K98 by a factor of 1/self.v. See Planck XIV
        # section 4.1.2.
        term3 = 1 + (ζ_over_δ_ζ**self.p) / self.v

        term4 = 1 - np.exp(-((R_band / self.delta_r) ** 20))

        return term1 * term2 * term3 * term4


@dataclass
class Ring(Component):
    """DIRBE Circum-solar Ring (excluding the Earth-trailing Feature).

    Parameters
    ----------
    n_0
        Density at 1 AU.
    R
        Radius of the peak density.
    sigma_r
        Radial dispersion.
    sigma_z
        Vertical dispersion.
    """

    n_0: float
    R: float
    sigma_r: float
    sigma_z: float

    def compute_density(
        self, X_helio: NDArray[np.floating], **_
    ) -> NDArray[np.floating]:
        """See base class for documentation."""

        X_ring = X_helio - self.X_0
        R_ring = np.sqrt(X_ring[0] ** 2 + X_ring[1] ** 2 + X_ring[2] ** 2)

        Z_ring = (
            X_ring[0] * self.sin_Omega_rad * self.sin_i_rad
            - X_ring[1] * self.cos_Omega_rad * self.sin_i_rad
            + X_ring[2] * self.cos_i_rad
        )
        # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
        # term. See Planck 2013 XIV, section 4.1.3.
        term1 = -((R_ring - self.R) ** 2) / self.sigma_r**2
        term2 = np.abs(Z_ring) / self.sigma_z

        return self.n_0 * np.exp(term1 - term2)


@dataclass
class Feature(Component):
    """DIRBE Earth-trailing Feature.

    Parameters
    ----------
    n_0
        Density at 1 AU.
    R
        Radius of the peak density.
    sigma_r
        Radial dispersion.
    sigma_z
        Vertical dispersion.
    theta
        Longitude with respect to Earth [deg].
    sigma_theta
        Longitude dispersion [deg].
    """

    n_0: float
    R: float
    sigma_r: float
    sigma_z: float
    theta: float
    sigma_theta: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.theta_rad = np.radians(self.theta)
        self.sigma_theta_rad = np.radians(self.sigma_theta)

    def compute_density(
        self,
        X_helio: NDArray[np.floating],
        X_earth: NDArray[np.floating],
        **_,
    ) -> NDArray[np.floating]:
        """See base class for documentation."""

        X_feature = X_helio - self.X_0
        R_feature = np.sqrt(X_feature[0] ** 2 + X_feature[1] ** 2 + X_feature[2] ** 2)

        Z_feature = (
            X_feature[0] * self.sin_Omega_rad * self.sin_i_rad
            - X_feature[1] * self.cos_Omega_rad * self.sin_i_rad
            + X_feature[2] * self.cos_i_rad
        )
        X_earth_comp = X_earth - self.X_0

        θ_comp = np.arctan2(X_feature[1], X_feature[0]) - np.arctan2(
            X_earth_comp[1], X_earth_comp[0]
        )

        Δθ = θ_comp - self.theta_rad
        condition1 = Δθ < -np.pi
        condition2 = Δθ > np.pi
        Δθ[condition1] = Δθ[condition1] + 2 * np.pi
        Δθ[condition2] = Δθ[condition2] - 2 * np.pi

        # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
        # term. See Planck 2013 XIV, section 4.1.3.
        exp_term = (R_feature - self.R) ** 2 / self.sigma_r**2
        exp_term += np.abs(Z_feature) / self.sigma_z
        exp_term += Δθ**2 / self.sigma_theta_rad**2

        return self.n_0 * np.exp(-exp_term)
