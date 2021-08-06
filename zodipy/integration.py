from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy

from zodipy import CompLabel


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
    def shells(self) -> np.ndarray:
        """Linearly spaced grid of distances from observer."""

        return np.linspace(self.R_min, self.R_max, self.n)

    @property
    def dshells(self) -> np.ndarray:
        """Distance between grid points in self.shells."""

        return np.diff(self.shells)


DEFAULT_CONFIG = {
    CompLabel.CLOUD: IntegrationConfig(
        R_min=0.0001, 
        R_max=30, 
        n=15,
        integrator=np.trapz
    ),
    CompLabel.BAND1: IntegrationConfig(
        R_min=0.25, 
        R_max=30, 
        n=15,
        integrator=np.trapz
    ),
    CompLabel.BAND2: IntegrationConfig(
        R_min=0.0001, 
        R_max=30, 
        n=15,
        integrator=np.trapz
    ),
    CompLabel.BAND3: IntegrationConfig(
        R_min=0.25, 
        R_max=30, 
        n=15,
        integrator=np.trapz
    ),
    CompLabel.RING: IntegrationConfig(
        R_min=0.0001, 
        R_max=2.5, 
        n=50,
        integrator=scipy.integrate.simpson
    ),
    CompLabel.FEATURE: IntegrationConfig(
        R_min=0.0001, 
        R_max=1, 
        n=15,
        integrator=scipy.integrate.simpson
    )
}
