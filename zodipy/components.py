from abc import ABC, abstractmethod

import numpy as np

from zodipy import component_params as params 


class BaseComponent(ABC):
    """Abstract base class for a Zodiacal component.
    
    It contains a method to get the coordinates of a shell around an 
    observer in the reference frame of the component. Additionally, any
    component that inherits from the class must implement a get_density
    method.
    """

    def __init__(self, parameters: params.BaseParameters) -> None:
        """Initializes a Zodical component.
        
        Parameters
        ----------
        parameters : `zodipy.component_params.BaseParameters`
            Parameter object containing all the parameters required to 
            evaluate the density of the component.
        """

        self.parameters = parameters

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
    
    def get_coordinates(self, X_observer, X_unit, R) -> tuple:
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

        # TODO: get proper earth coords        
        x_earth, y_earth = x0, y0

        x_helio = R*u_x + x0
        y_helio = R*u_y + y0
        z_helio = R*u_z + z0

        R_helio = np.sqrt(x_helio**2 + y_helio**2 + z_helio**2)

        x_prime = x_helio - self.parameters.x0
        y_prime = y_helio - self.parameters.y0
        z_prime = z_helio - self.parameters.z0

        omega = self.parameters.omega
        inclination = self.parameters.inclination

        R_prime = np.sqrt(x_prime**2 + y_prime**2 + z_prime**2)
        Z_prime = (
            x_prime*np.sin(omega)*np.sin(inclination) 
            - y_prime*np.cos(omega)*np.sin(inclination)
            + z_prime*np.cos(inclination)
        )

        theta_prime = np.arctan2(y_prime, x_prime)
        theta_earth = np.arctan2(y_earth, x_earth)
        theta = theta_prime - theta_earth

        # Constraining theta to be in the limit [-pi, pi]
        theta[theta < np.pi] = theta[theta < np.pi] + 2*np.pi
        theta[theta > np.pi] = theta[theta > np.pi] - 2*np.pi

        return (R_prime, Z_prime, theta), R_helio


class Cloud(BaseComponent):
    """The Zodiacal diffuse cloud component.

    This is a class representing the diffuse cloud in the K98 IPD model. 
    It provides a method to estimate the density of the diffuse cloud at 
    at a shell around the observer.
    """

    def __init__(self, parameters: params.CloudParameters) -> None:
        super().__init__(parameters)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        parameters = self.parameters
        zeta = np.abs(Z_prime) / R_prime
        mu = parameters.mu
        g = np.zeros_like(zeta)
        
        condition = zeta < mu
        g[condition] = zeta[condition]**2 / 2*mu
        g[~condition] = zeta[~condition] - (mu / 2)

        return (
            parameters.n0 * R_prime**-parameters.alpha 
            * np.exp(-parameters.beta * g * parameters.gamma)
        )


class Band(BaseComponent):
    """The Zodiacal astroidal band component.

    This is a class representing the astroidal dust band components in 
    the K98 IPD model. It provides a method to estimate the density of 
    the dust bands at at a shell around the observer.
    """

    def __init__(self, parameters: params.BandParameters) -> None:
        super().__init__(parameters)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        parameters = self.parameters
        zeta = np.abs(Z_prime) / R_prime
        zeta_over_delta_zeta = zeta/parameters.delta_zeta
        term1 = (3*parameters.n0/R_prime) * np.exp(-(zeta_over_delta_zeta)**6)
        term2 = 1 + ((zeta_over_delta_zeta)**parameters.p)/parameters.v
        term3 = 1 - np.exp(-(R_prime/parameters.delta_r)**20)
        
        return term1 * term2 * term3


class Ring(BaseComponent):
    """The Zodiacal circum-solar ring component.

    This is a class representing the circum-solar ring component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the circum-solar ring at at a shell around the observer.
    """

    def __init__(self, parameters: params.RingParameters) -> None:
        super().__init__(parameters)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        parameters = self.parameters
        term1 = -((R_prime - parameters.R)/parameters.sigma_r)**2
        term2 = np.abs(Z_prime) / parameters.sigma_z

        return parameters.n0 * np.exp(term1 - term2)


class Feature(BaseComponent):
    """The Zodiacal Earth-trailing feature component.

    This is a class representing the Earth-trailing feature component in 
    the K98 IPD model. It provides a method to estimate the density of 
    the Earth-trailing feature at at a shell around the observer.
    """

    def __init__(self, parameters: params.FeatureParameters) -> None:
        super().__init__(parameters)

    def get_density(self, R_prime, Z_prime, theta) -> np.ndarray:
        """See base class."""

        parameters = self.parameters
        term1 = -((R_prime - parameters.R)/parameters.sigma_r)**2
        term2 = np.abs(Z_prime) / parameters.sigma_z
        term3 = ((theta - parameters.theta) / parameters.sigma_theta)**2
        return parameters.n0 * np.exp(term1 - term2 - term3)