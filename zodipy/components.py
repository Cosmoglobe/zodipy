from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians, sin, cos
from math import pi as π
from typing import Tuple

import numpy as np

import zodipy._functions as F


@dataclass
class BaseComponent(ABC):
    """Abstract base class for a Zodiacal component.
    
    It contains a method to get the coordinates of a shell around an 
    observer in the reference frame of the component. Additionally, any
    component that inherits from the class must implement a get_density
    method.

    Parameters
    ----------
    x_0: float
        x offset from the Sun in ecliptic coordinates.
    y_0: float
        y offset from the Sun in ecliptic coordinates.
    z_0: float
        z offset from the Sun in ecliptic coordinates.
    inclination: float
        Inclination [deg].
    Ω: float
        Ascending node [deg].
    """

    x_0 : float
    y_0 : float
    z_0 : float
    i : float
    Ω : float

    def __post_init__(self) -> None:
        self.i = radians(self.i)
        self.Ω = radians(self.Ω)

    @abstractmethod
    def get_density(
        self, R_prime: np.ndarray, Z_prime: np.ndarray, θ: np.ndarray
    ) -> np.ndarray:
        """Returns the dust density at a shell around the observer.
        
        Parameters
        ----------
        R_prime : `numpy.ndarray`
            Array containing the distance to the coordinate where the
            density is being evaluated per pixel in the prime coordinate
            system. The shape is (`NPIX`).
        Z_prime : `numpy.ndarray`
            Array containing the height above the x-y-plane in the prime 
            coordinate system of the coordinate in R_prime. The shape is
            (`NPIX`).
        θ : `numpy.ndarray`
            Array containing the heliocentric ecliptic longitude of the 
            coords in R_prime releative to the longitude of Earth. The 
            shape is (`NPIX`).

        Returns
        -------
        `numpy.ndarray`
            Array containing the density of the component at a shell around
            the observer given by R_prime, Z_prime, and θ. The shape is
            (`NPIX`)
        """

    def get_coordinates(
        self, 
        X_observer: np.ndarray, 
        X_earth: np.ndarray, 
        X_unit: np.ndarray, 
        R: np.ndarray
    ) -> Tuple[np.ndarray]:
        """Returns coordinates for which to evaluate the density.
        
        The density of a component is computed in the prime coordinate 
        system, representing a coordinate system with the origin centered
        on the component.

        Parameters
        ----------
        X_observer : `numpy.ndarray`
            Vector containing to coordinates of the observer.
            The shape is (3,).
        X_unit : `numpy.ndarray`
            Array containing the unit vectors pointing to each pixel in 
            the HEALPIX map. The shape is (3, `NPIX`).
        R : `numpy.ndarray`
            Array containing grid distances to the surface of a shells 
            centered on the observer.

        Returns
        -------
        R_prime : `numpy.ndarray`
            Array containing the distance to the coordinate where the
            density is being evaluated per pixel in the prime coordinate
            system. The shape is (`NPIX`).
        Z_prime : `numpy.ndarray`
            Array containing the height above the x-y-plane in the prime 
            coordinate system of the coordinate in R_prime. The shape is
            (`NPIX`).
        θ : `numpy.ndarray`
            Array containing the heliocentric ecliptic longitude of the 
            coords in R_prime releative to the longitude of Earth. The 
            shape is (`NPIX`).
        R_helio : `np.ndarray`
            Array containing the heliocentric distance to the coordinate
            where the density is evaluated. The shape is (`NPIX`).
        """

        u_x, u_y, u_z = X_unit
        x_0, y_0, z_0 = X_observer
        
        x_helio = R*u_x + x_0
        y_helio = R*u_y + y_0
        z_helio = R*u_z + z_0
        R_helio = np.sqrt(x_helio**2 + y_helio**2 + z_helio**2)

        x_prime = x_helio - self.x_0
        y_prime = y_helio - self.y_0
        z_prime = z_helio - self.z_0

        Ω, i = self.Ω, self.i
        R_prime = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2)
        Z_prime = (
            x_prime*sin(Ω)*sin(i) 
            - y_prime*cos(Ω)*sin(i)
            + z_prime*cos(i)
        )

        x_earth, y_earth, _ = X_earth
        θ_prime = np.arctan2(y_prime, x_prime)
        θ_earth = np.arctan2(y_earth, x_earth)
        θ = θ_prime - θ_earth

        # Constraining θ to be in the limit [-π, π]
        θ[θ < π] = θ[θ < π] + 2*π
        θ[θ > π] = θ[θ > π] - 2*π

        return (R_prime, Z_prime, θ), R_helio

    def get_emission(self, freq: float, X_observer, X_earth, X_unit, R) -> np.ndarray:
        """Returns the emission at a shell of distance R from the observer.
        
        For a description on X_observer, X_earth, X_unit and R, please 
        see the get_coords function.

        Parameters
        ----------
        freq : float
            Frequency at which to evaluate the emitted emission.

        Returns
        -------
        emission : `np.ndarray`
            Array containing the Zodiacal emission from a component at 
            different shells around the observer. The shape is 
            (len(R), `NPIX`).
        """

        coords, R_helio = self.get_coordinates(X_observer, X_earth, X_unit, R)
        density = self.get_density(*coords)
        temperature = F.interplanetary_temperature(R_helio)
        blackbody_emission = F.blackbody_emission(temperature, freq)

        return blackbody_emission * density


@dataclass
class Cloud(BaseComponent):
    """The Zodiacal diffuse cloud component.

    This is a class representing the diffuse cloud in the K98 IPD model. 
    It provides a method to estimate the density of the diffuse cloud at 
    at a shell around the observer.

    Parameters
    ----------
    n_0 : float
        Density at 1 AU.
    α : float 
        Radial power-law exponent.
    β : float
        Vertical shape parameter.
    γ : float
        Vertical power-law exponent.
    μ : float
        Widening parameter for the modified fan.
    """

    n_0 : float
    α : float
    β : float
    γ : float
    μ : float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime, Z_prime, θ):
        """See base class for documentation."""

        ζ = np.abs(Z_prime) / R_prime
        μ = self.μ
        g = np.zeros_like(ζ)
        
        condition = ζ < μ
        g[condition] = ζ[condition]**2 / 2*μ
        g[~condition] = ζ[~condition] - (μ / 2)

        return (
            self.n_0 * R_prime**-self.α 
            * np.exp(-self.β * g * self.γ)
        )


@dataclass
class Band(BaseComponent):
    """The Zodiacal astroidal band component.

    This is a class representing the astroidal dust band components in 
    the K98 IPD model. It provides a method to estimate the density of 
    the dust bands at at a shell around the observer.

    Parameters
    ----------
    n_0 : float
        Density at 3 AU.
    δ_ζ : float
        Shape parameter [deg].
    v : float
        Shape parameter.
    p : float
        Shape parameter.
    δ_r : float
        Inner radial cutoff. 
    """

    n_0 : float
    δ_ζ : float
    v : float
    p : float
    δ_r : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.δ_ζ = radians(self.δ_ζ)

    def get_density(self, R_prime, Z_prime, θ):
        """See base class for documentation."""

        ζ = np.abs(Z_prime) / R_prime
        ζ_over_δ_ζ = ζ/self.δ_ζ
        term1 = (3*self.n_0/R_prime) * np.exp(-(ζ_over_δ_ζ)**6)
        term2 = 1 + ((ζ_over_δ_ζ)**self.p)/self.v
        term3 = 1 - np.exp(-(R_prime/self.δ_r)**20)
        
        return term1 * term2 * term3


@dataclass
class Ring(BaseComponent):
    """The Zodiacal circum-solar ring component.

    This is a class representing the circum-solar ring component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the circum-solar ring at at a shell around the observer.

    Parameters
    ----------
    n_0 : float
        Density at 1 AU.
    R : float
        Radius of the peak density.
    σ_r : float
        Radial dispersion.
    σ_z : float 
        Vertical dispersion.
    """

    n_0 : float
    R : float
    σ_r : float
    σ_z : float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime, Z_prime, θ):
        """See base class for documentation."""

        term1 = -((R_prime - self.R)/self.σ_r)**2
        term2 = np.abs(Z_prime) / self.σ_z

        return self.n_0 * np.exp(term1 - term2)


@dataclass
class Feature(BaseComponent):
    """The Zodiacal Earth-trailing feature component.

    This is a class representing the Earth-trailing feature component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the Earth-trailing feature at at a shell around the observer.

    Parameters
    ----------
    n_0 : float
        Density at 1 AU.
    R : float
        Radius of the peak density.
    σ_r : float
        Radial dispersion.
    σ_z : float 
        Vertical dispersion.
    θ : float
        Longitude with respect to Earth.
    σ_θ : float
        Longitude dispersion.
    """

    n_0 : float
    R : float
    σ_r : float
    σ_z : float
    θ : float
    σ_θ : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.θ = radians(self.θ)
        self.σ_θ = radians(self.σ_θ)

    def get_density(self, R_prime, Z_prime, θ):
        """See base class for documentation."""

        term1 = -((R_prime - self.R)/self.σ_r)**2
        term2 = np.abs(Z_prime) / self.σ_z
        term3 = ((θ - self.θ) / self.σ_θ)**2
        return self.n_0 * np.exp(term1 - term2 - term3)