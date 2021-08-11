from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy


@dataclass
class IntegrationConfig:
    """Config object for the LOS/shell integration in the simulation.
    
    Attributes
    ----------
    R_min : float
        Minimum distance from the observer for which to evaluate the 
        dust density.
    R_max : float 
        Maximum distance from the observer for which to evaluate the 
        dust density.
    n : int
        Number of shells for which to evaluate the dust density.
    integator : func
        Function which performs the integration.
    """

    R_min : float
    R_max : float
    n : int
    integrator : Callable

    @property
    def R(self) -> np.ndarray:
        """Linearly spaced grid of distances from observer."""

        return np.expand_dims(np.linspace(self.R_min, self.R_max, self.n), axis=1)

    @property
    def dR(self) -> np.ndarray:
        """Distance between grid points in self.shells."""

        return np.diff(self.R)


DEFAULT_CONFIG = {
    'cloud': IntegrationConfig(0.0001, 30, 15, np.trapz),
    'band1': IntegrationConfig(0.0001, 30, 15, np.trapz),
    'band2': IntegrationConfig(0.0001, 30, 15, np.trapz),
    'band3': IntegrationConfig(0.0001, 30, 15, np.trapz),
    'ring': IntegrationConfig(0.0001, 2.25, 50, scipy.integrate.simpson),
    'feature': IntegrationConfig(0.0001, 1, 15, scipy.integrate.simpson)
}