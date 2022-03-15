from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from astropy.units import Quantity
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

π = np.pi


@dataclass
class Component(ABC):
    """Base class for an Interplanetary Dust Component.

    Any component that inherits from this class needs to implement the two
    abstract methods `get_compcentric_coordinates` and `compute_density`.

    Parameters
    ----------
    x_0
        x-offset from the Sun in heliocentric ecliptic coordinates in AU.
    y_0
        y-offset from the Sun in heliocentric ecliptic coordinates in AU.
    z_0
        z-offset from the Sun in heliocentric ecliptic coordinates in AU.
    i
        Inclination with respect to the ecliptic planein deg.
    Omega
        Ascending node in deg.
    """

    x_0: Quantity[u.AU]
    y_0: Quantity[u.AU]
    z_0: Quantity[u.AU]
    i: Quantity[u.deg] | Quantity[u.rad]
    Omega: Quantity[u.deg] | Quantity[u.rad]

    def __post_init__(self) -> None:
        # [AU] -> [AU per 1 AU]
        self.x_0 = (self.x_0 / u.AU).value
        self.y_0 = (self.y_0 / u.AU).value
        self.z_0 = (self.z_0 / u.AU).value
        self.X_0 = np.expand_dims(np.asarray([self.x_0, self.y_0, self.z_0]), axis=1)

        self.i = self.i.to(u.rad).value
        self.Omega = self.Omega.to(u.rad).value

        # Computing frequently used variables
        self.sin_i = np.sin(self.i)
        self.cos_i = np.cos(self.i)
        self.sin_Omega = np.sin(self.Omega)
        self.cos_Omega = np.cos(self.Omega)

    @abstractmethod
    def compute_density(
        self,
        X_helio: NDArray[np.float_],
        *,
        X_earth: Optional[NDArray[np.float_]] = None,
        X_0_cloud: Optional[NDArray[np.float_]] = None,
    ) -> NDArray[np.float_]:
        """Returns the dust density of a component at points in the Solar System
        given by 'X_helio'.

        Parameters
        ----------
        X_helio
            Heliocentric ecliptic coordinates (x, y, z) of points in the Solar
            System.
        X_earth
            Heliocentric ecliptic coordinates of the Earth. Required for the
            Earth-trailing Feature-
        X_0_cloud
            Heliocentric ecliptic coordinates of the Diffuse Clouds offset.
            Required for the Dust Bands.

        Returns
        -------
            Density of the component at points in the Solar System.
        """


@dataclass
class Cloud(Component):
    """The Zodiacal Diffuse Cloud component.

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

    n_0: Quantity[u.AU ** -1]
    alpha: float
    beta: float
    gamma: float
    mu: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_0 = self.n_0.value

    def compute_density(self, X_helio: NDArray[np.float_], **_) -> NDArray[np.float_]:
        """See base class for documentation."""

        X_comp = X_helio - self.X_0
        R_comp = np.linalg.norm(X_comp, axis=0)

        Z_comp = (
            X_comp[0] * self.sin_Omega * self.sin_i
            - X_comp[1] * self.cos_Omega * self.sin_i
            + X_comp[2] * self.cos_i
        )

        ζ = np.abs(Z_comp / R_comp)
        μ = self.mu
        g = np.zeros_like(ζ)

        condition = ζ < μ
        g[condition] = ζ[condition] ** 2 / (2 * μ)
        g[~condition] = ζ[~condition] - (μ / 2)

        return self.n_0 * R_comp ** -self.alpha * np.exp(-self.beta * g ** self.gamma)


@dataclass
class Band(Component):
    """The Zodiacal Astroidal Band component.

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

    n_0: Quantity[u.AU ** -1]
    delta_zeta: Quantity[u.deg] | Quantity[u.rad]
    v: float
    p: float
    delta_r: Quantity[u.AU]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_0 = self.n_0.value
        self.delta_zeta = self.delta_zeta.to(u.rad)
        # NOTE: zeta/delta_zeta has to be unitless in the K98 migrating band
        # expression. We are unsure why the units of delta_zeta is given in
        # degrees, but it could be due to some small angle approximation with
        # cos theta. Nevertheless, we must divide away the units of radians
        # for the zeta/delta_zeta to be unitless.

        self.delta_zeta = (self.delta_zeta / u.rad).value
        self.delta_r = (self.delta_r / u.AU).value  # [AU] -> [AU per 1 AU]

    def compute_density(
        self, X_helio: NDArray[np.float_], X_0_cloud: NDArray[np.float_], **_
    ) -> NDArray[np.float_]:
        """See base class for documentation."""

        X_comp = X_helio - X_0_cloud
        R_comp = np.linalg.norm(X_comp, axis=0)

        Z_comp = (
            X_comp[0] * self.sin_Omega * self.sin_i
            - X_comp[1] * self.cos_Omega * self.sin_i
            + X_comp[2] * self.cos_i
        )

        ζ = np.abs(Z_comp / R_comp)
        ζ_over_δ_ζ = ζ / self.delta_zeta
        term1 = 3 * self.n_0 / R_comp
        term2 = np.exp(-(ζ_over_δ_ζ ** 6))

        # Differs from eq 8 in K98 by a factor of 1/self.v. See Planck XIV
        # section 4.1.2.
        term3 = 1 + (ζ_over_δ_ζ ** self.p) / self.v

        term4 = 1 - np.exp(-((R_comp / self.delta_r) ** 20))

        return term1 * term2 * term3 * term4


@dataclass
class Ring(Component):
    """The Zodiacal Circum-solar Ring component.

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

    n_0: Quantity[u.AU ** -1]
    R: Quantity[u.AU]
    sigma_r: Quantity[u.AU]
    sigma_z: Quantity[u.AU]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_0 = self.n_0.value

        # [AU] -> [AU per 1 AU]
        self.R = (self.R / u.AU).value
        self.sigma_r = (self.sigma_r / u.AU).value
        self.sigma_z = (self.sigma_z / u.AU).value

    def compute_density(
        self, X_helio: NDArray[np.float_], **_
    ) -> NDArray[np.float_]:
        """See base class for documentation."""

        X_comp = X_helio - self.X_0
        R_comp = np.linalg.norm(X_comp, axis=0)

        Z_comp = (
            X_comp[0] * self.sin_Omega * self.sin_i
            - X_comp[1] * self.cos_Omega * self.sin_i
            + X_comp[2] * self.cos_i
        )
        # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
        # term. See Planck 2013 XIV, section 4.1.3.
        term1 = -((R_comp - self.R) ** 2) / self.sigma_r ** 2
        term2 = np.abs(Z_comp) / self.sigma_z

        return self.n_0 * np.exp(term1 - term2)


@dataclass
class Feature(Component):
    """The Zodiacal Earth-trailing Feature component.

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
    θ
        Longitude with respect to Earth.
    sigma_θ
        Longitude dispersion.
    """

    n_0: Quantity[u.AU ** -1]
    R: Quantity[u.AU]
    sigma_r: Quantity[u.AU]
    sigma_z: Quantity[u.AU]
    theta: Quantity[u.deg]
    sigma_theta: Quantity[u.deg]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.n_0 = self.n_0.value
        self.theta = (self.theta.to(u.rad)).value
        self.sigma_theta = (self.sigma_theta.to(u.rad)).value

        # [AU] -> [AU per 1 AU]
        self.R = (self.R / u.AU).value
        self.sigma_r = (self.sigma_r / u.AU).value
        self.sigma_z = (self.sigma_z / u.AU).value

    def compute_density(
        self,
        X_helio: NDArray[np.float_],
        X_earth: NDArray[np.float_],
        **_,
    ) -> NDArray[np.float_]:
        """See base class for documentation."""

        X_comp = X_helio - self.X_0
        R_comp = np.linalg.norm(X_comp, axis=0)

        Z_comp = (
            X_comp[0] * self.sin_Omega * self.sin_i
            - X_comp[1] * self.cos_Omega * self.sin_i
            + X_comp[2] * self.cos_i
        )
        X_earth_comp = X_earth - self.X_0

        θ_comp = np.arctan2(X_comp[1], X_comp[0]) - np.arctan2(
            X_earth_comp[1], X_earth_comp[0]
        )

        Δθ = θ_comp - self.theta
        condition1 = Δθ < -π
        condition2 = Δθ > π
        Δθ[condition1] = Δθ[condition1] + 2 * π
        Δθ[condition2] = Δθ[condition2] - 2 * π

        # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
        # term. See Planck 2013 XIV, section 4.1.3.
        exp_term = (R_comp - self.R) ** 2 / self.sigma_r ** 2
        exp_term += np.abs(Z_comp) / self.sigma_z
        exp_term += Δθ ** 2 / self.sigma_theta ** 2

        return self.n_0 * np.exp(-exp_term)
