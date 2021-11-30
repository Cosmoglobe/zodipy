from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, sin, cos
from math import pi as π
from typing import Tuple, Union

from numba import njit
import numpy as np

from zodipy._functions import interplanetary_temperature, blackbody_emission


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
    Ω
        Ascending node [deg].
    """

    x_0: float
    y_0: float
    z_0: float
    i: float
    Ω: float

    def __post_init__(self) -> None:
        self.i = radians(self.i)
        self.Ω = radians(self.Ω)
        self.X_component = np.expand_dims([self.x_0, self.y_0, self.z_0], axis=1)

    @abstractmethod
    def get_density(
        self, R_prime: np.ndarray, Z_prime: np.ndarray, *, θ_prime: np.ndarray
    ) -> np.ndarray:
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

    @staticmethod
    @njit
    def get_coordinates(
        R_comp: Union[float, np.ndarray],
        X_observer: np.ndarray,
        X_earth: np.ndarray,
        X_unit: np.ndarray,
        X_component: np.ndarray,
        Ω_component: float,
        i_component: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns a set of coordinates for a component.

        Given a set of heliocentric ecliptic positions in space given by the
        unit vectors `X_unit` and `R_comp`, we compute the `R_helio`, `R_prime`,
        `Z_prime`, and `θ_prime` coordinates as seen by and observer whos
        heliocentric ecliptic coordinates are given by `X_observer`. These are
        the coordinates required by the Interplanetary Dust Model to evalutate
        the density of a Zodiacal Component at the given positions.

        Parameters
        ----------
        R_comp
            Distance R to a shell centered on the observer at which we want
            to evaluate the Zodiacal emission at.
        X_observer
            Heliocentric ecliptic cartesian coordinates of the observer.
        X_earth
            Heliocentric ecliptic cartesian coordinates of the Earth.
        X_unit
            Heliocentric ecliptic cartesian unit vectors pointing to each 
            position in space we that we consider.
        X_component
            Heliocentric ecliptic cartesian off-set of the component
            (x_0, y_0, z_0).
        Ω_component
            Ascending node of the component.
        i_component
            Inclination of the component.

        Returns
        -------
        R_helio
            Array of distances corresponding to discrete points along a
            line-of-sight for a shell surrounding the observer in heliocentric
            ecliptic coordinates.
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

        X_helio = R_comp * X_unit + X_observer
        R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

        X_prime = X_helio - X_component
        R_prime = np.sqrt(X_prime[0] ** 2 + X_prime[1] ** 2 + X_prime[2] ** 2)

        Z_prime = (
            X_prime[0] * sin(Ω_component) * sin(i_component)
            - X_prime[1] * cos(Ω_component) * sin(i_component)
            + X_prime[2] * cos(i_component)
        )

        X_earth_prime = X_earth - X_component[0]
        θ_prime = np.arctan2(X_prime[1], X_prime[0]) - np.arctan2(
            X_earth_prime[1], X_earth_prime[0]
        )

        return R_helio, R_prime, Z_prime, θ_prime

    def get_emission(
        self,
        distance_to_shell: Union[float, np.ndarray],
        observer_position: np.ndarray,
        earth_position: np.ndarray,
        unit_vectors: np.ndarray,
        freq: float,
    ) -> np.ndarray:
        """Returns the emission at a shell of distance R from the observer.

        For a description on X_observer, X_earth, X_unit and R, please
        see the get_coords function.

        Parameters
        ----------
        distance_to_shell
            Distance R to a shell centered on the observer for which we want
            to evaluate the Zodiacal emission.
        observer_position
            Heliocentric ecliptic cartesian coordinates of the observer.
        earth_position
            Heliocentric ecliptic cartesian coordinates of the Earth.
        unit_vectors
            Heliocentric ecliptic cartesian unit vectors pointing to each 
            position in space we that we consider.
        freq
            Frequency at which to evaluate the Zodiacal emission.

        Returns
        -------
        emission
            Zodiacal emission at
            Array containing the Zodiacal emission emitted from a shell at
            distance R from the observer. The shape is (len(R), `NPIX`).
        """

        observer_position = np.expand_dims(observer_position, axis=1)
        R_helio, R_prime, Z_prime, θ_prime = self.get_coordinates(
            R_comp=distance_to_shell,
            X_observer=observer_position,
            X_earth=earth_position,
            X_unit=unit_vectors,
            X_component=self.X_component,
            Ω_component=self.Ω,
            i_component=self.i,
        )
        density = self.get_density(
            R_prime=R_prime,
            Z_prime=Z_prime,
            θ_prime=θ_prime,
        )
        temperature = interplanetary_temperature(R=R_helio)
        emission = blackbody_emission(T=temperature, ν=freq)

        return emission * density


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
    α
        Radial power-law exponent.
    β
        Vertical shape parameter.
    γ
        Vertical power-law exponent.
    μ
        Widening parameter for the modified fan.
    """

    n_0: float
    α: float
    β: float
    γ: float
    μ: float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime: np.ndarray, Z_prime: np.ndarray, **_) -> np.ndarray:
        """See base class for documentation."""

        ζ = np.abs(Z_prime) / R_prime
        μ = self.μ
        g = np.zeros_like(ζ)

        condition = ζ < μ
        g[condition] = ζ[condition] ** 2 / (2 * μ)
        g[~condition] = ζ[~condition] - (μ / 2)

        return self.n_0 * R_prime ** -self.α * np.exp(-self.β * g * self.γ)


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
    δ_ζ
        Shape parameter [deg].
    v
        Shape parameter.
    p
        Shape parameter.
    δ_r
        Inner radial cutoff.
    """

    n_0: float
    δ_ζ: float
    v: float
    p: float
    δ_r: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.δ_ζ = radians(self.δ_ζ)

    def get_density(self, R_prime: np.ndarray, Z_prime: np.ndarray, **_) -> np.ndarray:
        """See base class for documentation."""

        ζ = np.abs(Z_prime) / R_prime
        ζ_over_δ_ζ = ζ / self.δ_ζ
        term1 = (3 * self.n_0 / R_prime) * np.exp(-((ζ_over_δ_ζ) ** 6))
        term2 = 1 + ((ζ_over_δ_ζ) ** self.p) / self.v
        term3 = 1 - np.exp(-((R_prime / self.δ_r) ** 20))

        return term1 * term2 * term3


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
    σ_r
        Radial dispersion.
    σ_z
        Vertical dispersion.
    """

    n_0: float
    R: float
    σ_r: float
    σ_z: float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime: np.ndarray, Z_prime: np.ndarray, **_) -> np.ndarray:
        """See base class for documentation."""

        term1 = -(((R_prime - self.R) / self.σ_r) ** 2)
        term2 = np.abs(Z_prime) / self.σ_z

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
    σ_r
        Radial dispersion.
    σ_z
        Vertical dispersion.
    θ
        Longitude with respect to Earth.
    σ_θ
        Longitude dispersion.
    """

    n_0: float
    R: float
    σ_r: float
    σ_z: float
    θ: float
    σ_θ: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.θ = radians(self.θ)
        self.σ_θ = radians(self.σ_θ)

    def get_density(
        self, R_prime: np.ndarray, Z_prime: np.ndarray, θ_prime: np.ndarray
    ) -> np.ndarray:
        """See base class for documentation."""

        Δθ = θ_prime - self.θ

        condition1 = Δθ < -π
        condition2 = Δθ > π
        Δθ[condition1] = Δθ[condition1] + 2 * π
        Δθ[condition2] = Δθ[condition2] - 2 * π

        term1 = -(((R_prime - self.R) / self.σ_r) ** 2)
        term2 = np.abs(Z_prime) / self.σ_z
        term3 = (Δθ / self.σ_θ) ** 2

        return self.n_0 * np.exp(term1 - term2 - term3)
