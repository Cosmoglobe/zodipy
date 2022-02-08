from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, sin, cos
from math import pi as π
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class Component(ABC):
    """Base class for an Interplanetary Dust Component.

    This class defines a method for getting the coordinates to a shell at
    distance R around an observer in the primed coordinate system
    (component-centric ecliptic cartesian coordinates).

    Attributes
    ----------
    x_0
        x-offset from the Sun in helliocentric ecliptic cartesian coordinates.
    y_0
        y-offset from the Sun in helliocentric ecliptic cartesian coordinates.
    z_0
        z-offset from the Sun in helliocentric ecliptic cartesian coordinates.
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
        self.i = radians(self.i)
        self.Omega = radians(self.Omega)
        self.X_0 = np.expand_dims([self.x_0, self.y_0, self.z_0], axis=1)

    @abstractmethod
    def compute_density(
        self,
        R_prime: NDArray[np.float64],
        Z_prime: NDArray[np.float64],
        *,
        θ_prime: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Returns the dust density at a shell around the observer.

        Parameters
        ----------
        R_prime
            Array of distances corresponding to discrete points along a
            line-of-sight for a shell surrounding the observer in the primed
            coordinates.
        Z_prime
            Heights above the midplane in primed coordinates of a component
            corresponding to the distances in `R_prime`.
        θ_prime
            Relative mean lognitude between the discrete points along the
            line-of-sight describe by `R_prime` and the Earth.

        Returns
        -------
            Density of the component at the coordinates given by R_prime,
            Z_prime, and θ_prime.
        """

    def get_primed_coordinates(
        self, X_helio: NDArray[np.float64], X_earth: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Returns a set of coordinates for a component.

        Given a set of heliocentric ecliptic positions in space given by the
        unit vectors `X_unit` and `R_comp`, we compute the `R_helio`, `R_prime`,
        `Z_prime`, and `θ_prime` coordinates as seen by and observer whos
        heliocentric ecliptic coordinates are given by `X_observer`. These are
        the coordinates required by the Interplanetary Dust Model to evalutate
        the density of a Zodiacal Component at the given positions.

        Parameters
        ----------
        X_helio
            Heliocentric ecliptic cartesian coordinates of each considered
            pixel.
        X_earth
            Heliocentric ecliptic cartesian coordinates of the Earth.

        Returns
        -------
        R_prime
            Array of distances corresponding to discrete points along a
            line-of-sight for a shell surrounding the observer in the primed
            coordinates.
        Z_prime
            Heights above the midplane in primed coordinates of a component
            corresponding to the distances in `R_prime`.
        θ_prime
            Relative mean lognitude between the discrete points along the
            line-of-sight describe by `R_prime` and the Earth.
        """

        X_prime = X_helio - self.X_0
        R_prime = np.linalg.norm(X_prime, axis=0)

        Z_prime = (
            X_prime[0] * sin(self.Omega) * sin(self.i)
            - X_prime[1] * cos(self.Omega) * sin(self.i)
            + X_prime[2] * cos(self.i)
        )

        X_earth_prime = np.expand_dims(X_earth, axis=1) - self.X_0
        θ_prime = np.arctan2(X_prime[1], X_prime[0]) - np.arctan2(
            X_earth_prime[1], X_earth_prime[0]
        )

        return R_prime, Z_prime, θ_prime

    def get_density(
        self,
        pixel_pos: NDArray[np.float64],
        earth_pos: NDArray[np.float64],
        cloud_offset: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Returns the component density at a shell around the observer.

        Parameters
        ----------
        pixel_pos
            Heliocentric ecliptic cartesian coordinates of the considered
            pixels.
        earth_pos
            Heliocentric ecliptic cartesian coordinates of the Earth.

        Returns
        -------
            Density of a Zodiacal component at a shell around the observer.
        """

        R_prime, Z_prime, θ_prime = self.get_primed_coordinates(
            X_helio=pixel_pos, X_earth=earth_pos
        )

        if isinstance(self, Band):
            X_cloud = pixel_pos - cloud_offset
            R_cloud = np.linalg.norm(X_cloud, axis=0)
            R_prime = R_cloud

        return self.compute_density(
            R_prime=R_prime,
            Z_prime=Z_prime,
            θ_prime=θ_prime,
        )


@dataclass
class Cloud(Component):
    """The Zodiacal Diffuse Cloud component.

    This class represents the diffuse cloud in the K98 IPD model. It provides a
    method to estimate the density of the diffuse cloud at a shell around the
    observer.

    Attributes
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

    def __post_init__(self) -> None:
        super().__post_init__()

    def compute_density(
        self, R_prime: NDArray[np.float64], Z_prime: NDArray[np.float64], **_
    ) -> NDArray[np.float64]:
        """See base class for documentation."""

        ζ = np.abs(Z_prime / R_prime)
        μ = self.mu
        g = np.zeros_like(ζ)

        condition = ζ < μ
        g[condition] = ζ[condition] ** 2 / (2 * μ)
        g[~condition] = ζ[~condition] - (μ / 2)

        return self.n_0 * R_prime ** -self.alpha * np.exp(-self.beta * g ** self.gamma)


@dataclass
class Band(Component):
    """The Zodiacal Astroidal Band component.

    This class represents the Astroidal Dust Band components in the K98 IPD
    model. It provides a method to estimate the density of the dust bands at a
    shell around the observer.

    Attributes
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
        self.delta_zeta = radians(self.delta_zeta)

    def compute_density(
        self,
        R_prime: NDArray[np.float64],
        Z_prime: NDArray[np.float64],
        **_,
    ) -> NDArray[np.float64]:
        """See base class for documentation."""

        ζ = np.abs(Z_prime / R_prime)
        ζ_over_δ_ζ = ζ / self.delta_zeta
        term1 = 3 * self.n_0 / R_prime
        term2 = np.exp(-(ζ_over_δ_ζ ** 6))
        term3 = self.v + ζ_over_δ_ζ ** self.p
        term4 = 1 - np.exp(-((R_prime / self.delta_r) ** 20))

        return term1 * term2 * term3 * term4


@dataclass
class Ring(Component):
    """The Zodiacal Circum-solar Ring component.

    This class represents the Circum-solar Ring component in the K98 IPD model.
    It provides a method to estimate the density of the Circum-solar Ring at a
    shell around the observer.

    Attributes
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

    def __post_init__(self) -> None:
        super().__post_init__()

    def compute_density(
        self, R_prime: NDArray[np.float64], Z_prime: NDArray[np.float64], **_
    ) -> NDArray[np.float64]:
        """See base class for documentation."""

        term1 = -((R_prime - self.R) ** 2) / (2 * self.sigma_r ** 2)
        term2 = np.abs(Z_prime) / self.sigma_z

        return self.n_0 * np.exp(term1 - term2)


@dataclass
class Feature(Component):
    """The Zodiacal Earth-trailing Feature component.

    This class represents the Earth-trailing Feature component in the K98 IPD
    model. It provides a method to estimate the density of the Earth-trailing
    Feature at a shell around the observer.

    Attributes
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

    n_0: float
    R: float
    sigma_r: float
    sigma_z: float
    theta: float
    sigma_theta: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.theta = radians(self.theta)
        self.sigma_theta = radians(self.sigma_theta)

    def compute_density(
        self,
        R_prime: NDArray[np.float64],
        Z_prime: NDArray[np.float64],
        θ_prime: NDArray[np.float64],
        **_,
    ) -> NDArray[np.float64]:
        """See base class for documentation."""

        Δθ = θ_prime - self.theta

        condition1 = Δθ < -π
        condition2 = Δθ > π
        Δθ[condition1] = Δθ[condition1] + 2 * π
        Δθ[condition2] = Δθ[condition2] - 2 * π

        term1 = -((R_prime - self.R) ** 2) / (2 * self.sigma_r ** 2)
        term2 = np.abs(Z_prime) / self.sigma_z
        term3 = Δθ ** 2 / (2 * self.sigma_theta ** 2)

        return self.n_0 * np.exp(term1 - term2 - term3)
