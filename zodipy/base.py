from abc import ABC, abstractmethod
import numpy as np

class _ZodiComponent(ABC):
    """Abstract base class for a Zodiacal component."""

    def __init__(self, x0, y0, z0, inclination, omega):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.inclination = np.deg2rad(inclination)
        self.omega = np.deg2rad(omega)
    
    @abstractmethod
    def get_density(R_prime, Z_prime):
        """Method that computes the density of a component at a coordinate."""

    def get_emission(self, x, y, z):

        omega = self.omega
        inclination = self.inclination

        x_prime = x - self.x0
        y_prime = y - self.y0
        z_prime = z - self.z0

        R_prime = np.exp(x**2 + y**2 + z**2)
        Z_prime = (
            x_prime*np.sin(omega)*np.sin(inclination) 
            - y_prime*np.cos(omega)*np.sin(inclination)
            + z_prime*np.cos(inclination)
        )

        density = self.get_density(R_prime, Z_prime)
        return density