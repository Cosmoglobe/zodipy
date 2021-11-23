from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, sin, cos
from math import pi as π
from typing import Tuple

from numba import njit
import numpy as np

from zodipy._functions import interplanetary_temperature, blackbody_emission


@dataclass
class Component(ABC):
    """Base class for an interplanetary dust component.

    This class contains a method that gets the coordinates of a shell
    around an observer in the reference frame of the component
    (prime coordinates).

    Any component that inherits from the class must implement a
    `get_density` method.

    Attributes
    ----------
    x_0
        x offset from the Sun in ecliptic coordinates.
    y_0
        y offset from the Sun in ecliptic coordinates.
    z_0
        z offset from the Sun in ecliptic coordinates.
    i
        Inclination [deg].
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
            Array containing the distance to the coordinate where the
            density is being evaluated per pixel in the prime coordinate
            system. The shape is (`NPIX`).
        Z_prime
            Array containing the height above the x-y-plane in the prime
            coordinate system of the coordinate in R_prime. The shape is
            (`NPIX`).
        θ_prime
            Array containing the heliocentric ecliptic longitude of the
            coords in R_prime releative to the longitude of Earth. The
            shape is (`NPIX`).

        Returns
        -------
            Array containing the density of the component at a shell around
            the observer given by R_prime, Z_prime, and θ. The shape is
            (`NPIX`)
        """

    @staticmethod
    @njit
    def get_coordinates(
        R_comp: np.ndarray,
        X_observer: np.ndarray,
        X_earth: np.ndarray,
        X_unit: np.ndarray,
        X_component: np.ndarray,
        Ω_component: float,
        i_component: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns the primed coordinates of a component.

        The density of a component is computed in the prime coordinate
        system, representing a coordinate system where the origin is
        centered on the component.

        NOTE: If the same line-of-sight vector R is used for all components,
        then the computation of the heliocentric coordinates X_helio and
        R_helio could be moved out of this function instead passed as arguments.
        This would mean that we only compute these `n_LOS` times instead of
        `n_LOS_comp´ * `n_comps` as we do now. However, we find that using a
        single line-of-sight vector to be wastefull in the case where we want
        to evaluate the earth-neighbouring components, which are only valid around
        ~1 AU.

        Parameters
        ----------
        R_comp
            Distance R to a shell centered on the observer at which we want
            to evaluate the Zodiacal emission at. Different shells are used
            for each component.
        X_observer
            Vector containing the coordinates of the observer.
            The shape is (3,).
        X_earth
            Array containing the heliocentric earth cooridnates. The shape
            is (3,).
        X_unit
            Array containing the unit vectors pointing to each pixel in
            the HEALPIX map. The shape is (3, `NPIX`).
        X_component
            Array containing the heliocentric cooridnates to the center of the
            component offset by (x0, y0, z0). The shape is (3,).
        Ω_component
            Ascending node of the component.
        i_component
            Inclination of the component.

        Returns
        -------
        R_helio
            Array containing the heliocentric distance to the coordinate
            where the density is evaluated. The shape is (`NPIX`).
        R_prime
            Array containing the distance to the coordinate where the
            density is being evaluated per pixel in the prime coordinate
            system. The shape is (`NPIX`).
        Z_prime
            Array containing the height above the x-y-plane in the prime
            coordinate system of the coordinate in R_prime. The shape is
            (`NPIX`).
        θ_prime
            Array containing the heliocentric ecliptic longitude of the
            coords in R_prime releative to the longitude of Earth. The
            shape is (`NPIX`).
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
        θ_prime = np.arctan2(X_prime[1], X_prime[2]) - np.arctan2(
            X_earth_prime[1], X_earth_prime[0]
        )

        return R_helio, R_prime, Z_prime, θ_prime

    def get_emission(
        self,
        distance_to_shell: np.ndarray,
        observer_coordinates: np.ndarray,
        earth_coordinates: np.ndarray,
        unit_vectors: np.ndarray,
        freq: float,
    ) -> np.ndarray:
        """Returns the emission at a shell of distance R from the observer.

        For a description on X_observer, X_earth, X_unit and R, please
        see the get_coords function.

        Parameters
        ----------
        distance_to_shell
            Distance R to a shell centered on the observer at which we want
            to evaluate the Zodiacal emission at.
        observer_coordinates
            Vector containing the coordinates of the observer.
            The shape is (3,).
        earth_coordinates
            Vector containing the coordinates of the Earth.
            The shape is (3,).
        unit_vectors
            Array containing the unit vectors pointing to each pixel in
            the HEALPIX map. The shape is (3, `NPIX`).
        freq
            Frequency at which to evaluate the emitted emission.

        Returns
        -------
        emission
            Array containing the Zodiacal emission emitted from a shell at
            distance R from the observer. The shape is (len(R), `NPIX`).
        """

        observer_coordinates = np.expand_dims(observer_coordinates, axis=1)

        R_helio, R_prime, Z_prime, θ_prime = self.get_coordinates(
            R_comp=distance_to_shell,
            X_observer=observer_coordinates,
            X_earth=earth_coordinates,
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
        temperature = interplanetary_temperature(R_helio)
        emission = blackbody_emission(temperature, freq)

        return emission * density


@dataclass
class Cloud(Component):
    """The Zodiacal diffuse cloud component.

    This is a class representing the diffuse cloud in the K98 IPD model.
    It provides a method to estimate the density of the diffuse cloud at
    at a shell around the observer.

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
        g[condition] = ζ[condition] ** 2 / 2 * μ
        g[~condition] = ζ[~condition] - (μ / 2)

        return self.n_0 * R_prime ** -self.α * np.exp(-self.β * g * self.γ)


@dataclass
class Band(Component):
    """The Zodiacal astroidal band component.

    This is a class representing the astroidal dust band components in
    the K98 IPD model. It provides a method to estimate the density of
    the dust bands at at a shell around the observer.

    Parameters
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
    """The Zodiacal circum-solar ring component.

    This is a class representing the circum-solar ring component in
    the K98 IPD model. It provides a method to estimate the density of
    the circum-solar ring at at a shell around the observer.

    Parameters
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
    """The Zodiacal Earth-trailing feature component.

    This is a class representing the Earth-trailing feature component in
    the K98 IPD model. It provides a method to estimate the density of
    the Earth-trailing feature at at a shell around the observer.

    Parameters
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
