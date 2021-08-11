from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    x0: float
        x offset from the Sun in ecliptic coordinates.
    y0: float
        y offset from the Sun in ecliptic coordinates.
    z0: float
        z offset from the Sun in ecliptic coordinates.
    inclination: float
        Inclination [deg].
    omega: float
        Ascending node [deg].
    """

    x0 : float
    y0 : float
    z0 : float
    inclination : float
    omega : float

    def __post_init__(self) -> None:
        self.inclination = np.deg2rad(self.inclination)
        self.omega = np.deg2rad(self.omega)

    @abstractmethod
    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
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
        theta : `numpy.ndarray`
            Array containing the heliocentric ecliptic longitude of the 
            coords in R_prime releative to the longitude of Earth. The 
            shape is (`NPIX`).

        Returns
        -------
        `numpy.ndarray`
            Array containing the density of the component at a shell around
            the observer given by R_prime, Z_prime, and theta. The shape is
            (`NPIX`)
        """
    
    def get_coordinates(self, X_observer, X_earth, X_unit, R) -> Tuple[np.ndarray]:
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
        R : float
            Distance to the surface of a shell centered on the observer.

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
        theta : `numpy.ndarray`
            Array containing the heliocentric ecliptic longitude of the 
            coords in R_prime releative to the longitude of Earth. The 
            shape is (`NPIX`).
        R_helio : `np.ndarray`
            Array containing the heliocentric distance to the coordinate
            where the density is evaluated. The shape is (`NPIX`).
        """

        u_x, u_y, u_z = X_unit
        x0, y0, z0 = X_observer

        x_helio = R*u_x + x0
        y_helio = R*u_y + y0
        z_helio = R*u_z + z0
        R_helio = np.sqrt(x_helio**2 + y_helio**2 + z_helio**2)

        x_prime = x_helio - self.x0
        y_prime = y_helio - self.y0
        z_prime = z_helio - self.z0

        omega, inclination = self.omega, self.inclination
        R_prime = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2)
        Z_prime = (
            x_prime*np.sin(omega)*np.sin(inclination) 
            - y_prime*np.cos(omega)*np.sin(inclination)
            + z_prime*np.cos(inclination)
        )

        x_earth, y_earth, _ = X_earth
        theta_prime = np.arctan2(y_prime, x_prime)
        theta_earth = np.arctan2(y_earth, x_earth)
        theta = theta_prime - theta_earth

        # Constraining theta to be in the limit [-pi, pi]
        theta[theta < np.pi] = theta[theta < np.pi] + 2*np.pi
        theta[theta > np.pi] = theta[theta > np.pi] - 2*np.pi

        return (R_prime, Z_prime, theta), R_helio

    def get_emission(self, freq, X_observer, X_earth, X_unit, R) -> np.ndarray:
        """Returns the emission at a shell of distance R from the observer.
        
        See get_coordinates for a description of the parameters not described
        here.

        Parameters
        ----------
        freq : float
            Frequency at which to evaluate the emitted emission.

        Returns
        -------
        emission : `np.ndarray`
            Zodiacal emission from a component.
        """

        coordinates, R = self.get_coordinates(X_observer, X_earth, X_unit, R)
        density = self.get_density(*coordinates)
        temperature = F.interplanetary_temperature(R)
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
    n0 : float
        Density at 1 AU.
    alpha : float 
        Radial power-law exponent.
    beta : float
        Vertical shape parameter.
    gamma : float
        Vertical power-law exponent.
    mu : float
        Widening parameter for the modified fan.
    """

    n0 : float
    alpha : float
    beta : float
    gamma : float
    mu : float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        zeta = np.abs(Z_prime) / R_prime
        mu = self.mu
        g = np.zeros_like(zeta)
        
        condition = zeta < mu
        g[condition] = zeta[condition]**2 / 2*mu
        g[~condition] = zeta[~condition] - (mu / 2)

        return (
            self.n0 * R_prime**-self.alpha 
            * np.exp(-self.beta * g * self.gamma)
        )


@dataclass
class Band(BaseComponent):
    """The Zodiacal astroidal band component.

    This is a class representing the astroidal dust band components in 
    the K98 IPD model. It provides a method to estimate the density of 
    the dust bands at at a shell around the observer.

    Parameters
    ----------
    n0 : float
        Density at 3 AU.
    delta_zeta : float
        Shape parameter [deg].
    v : float
        Shape parameter.
    p : float
        Shape parameter.
    delta_r : float
        Inner radial cutoff. 
    """

    n0 : float
    delta_zeta : float
    v : float
    p : float
    delta_r : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.delta_zeta = np.deg2rad(self.delta_zeta)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        zeta = np.abs(Z_prime) / R_prime
        zeta_over_delta_zeta = zeta/self.delta_zeta
        term1 = (3*self.n0/R_prime) * np.exp(-(zeta_over_delta_zeta)**6)
        term2 = 1 + ((zeta_over_delta_zeta)**self.p)/self.v
        term3 = 1 - np.exp(-(R_prime/self.delta_r)**20)
        
        return term1 * term2 * term3


@dataclass
class Ring(BaseComponent):
    """The Zodiacal circum-solar ring component.

    This is a class representing the circum-solar ring component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the circum-solar ring at at a shell around the observer.

    Parameters
    ----------
    n0 : float
        Density at 1 AU.
    R : float
        Radius of the peak density.
    sigma_r : float
        Radial dispersion.
    sigma_z : float 
        Vertical dispersion.
    """

    n0 : float
    R : float
    sigma_r : float
    sigma_z : float

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        term1 = -((R_prime - self.R)/self.sigma_r)**2
        term2 = np.abs(Z_prime) / self.sigma_z

        return self.n0 * np.exp(term1 - term2)


@dataclass
class Feature(BaseComponent):
    """The Zodiacal Earth-trailing feature component.

    This is a class representing the Earth-trailing feature component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the Earth-trailing feature at at a shell around the observer.

    Parameters
    ----------
    n0 : float
        Density at 1 AU.
    R : float
        Radius of the peak density.
    sigma_r : float
        Radial dispersion.
    sigma_z : float 
        Vertical dispersion.
    theta : float
        Longitude with respect to Earth.
    sigma_theta : float
        Longitude dispersion.
    """

    n0 : float
    R : float
    sigma_r : float
    sigma_z : float
    theta : float
    sigma_theta : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.theta = np.deg2rad(self.theta)
        self.sigma_theta = np.deg2rad(self.sigma_theta)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        term1 = -((R_prime - self.R)/self.sigma_r)**2
        term2 = np.abs(Z_prime) / self.sigma_z
        term3 = ((theta - self.theta) / self.sigma_theta)**2
        return self.n0 * np.exp(term1 - term2 - term3)