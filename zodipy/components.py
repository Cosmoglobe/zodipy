import numpy as np
from zodipy.base import _ZodiComponent

class Cloud(_ZodiComponent):
    """The Diffuse Cloud component."""
    def __init__(
        self, 
        x0, 
        y0, 
        z0, 
        inclination, 
        omega, 
        n0,
        alpha,
        beta,
        gamma,
        mu
    ):
        super().__init__(x0, y0, z0, inclination, omega)
        self.n0 = n0
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mu = mu

    def get_density(self, R_prime, Z_prime):
        """Get density of component at some radial distance R and vertical 
        height Z above the ecliptic.
        """

        zeta = np.abs(Z_prime) / R_prime
        mu = self.mu
        if zeta < (mu := self.mu):
            g = zeta**2 / 2*mu
        else:
            g = zeta - (mu / 2)
    
        return self.n0 * R_prime**-self.alpha * np.exp(-self.beta * g * self.gamma)