from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy


@dataclass
class IntegrationConfig:
    """Config object for the LOS/shell integration in the simulation.
    
    Attributes
    ----------
    R_max
        Maximum distance from the observer for which to evaluate the 
        dust density.
    n
        Number of shells for which to evaluate the dust density.
    integator
        Function which performs the integration.
    """

    R_max : float
    n : int
    integrator : Callable[[float, float, int], np.ndarray] = np.trapz

    @property
    def R(self) -> np.ndarray:
        """Linearly spaced grid of distances to shells around the observer."""

        ZERO = np.finfo(float).eps
        return np.expand_dims(np.linspace(ZERO, self.R_max, self.n), axis=1)

    @property
    def dR(self) -> np.ndarray:
        """Distance between grid points in R"""

        return np.diff(self.R)


_CUT_OFF = 6

DEFAULT = {
    'cloud': IntegrationConfig(R_max=_CUT_OFF, n=250),
    'band1': IntegrationConfig(R_max=_CUT_OFF, n=50),
    'band2': IntegrationConfig(R_max=_CUT_OFF, n=50),
    'band3': IntegrationConfig(R_max=_CUT_OFF, n=50),
    'ring': IntegrationConfig(R_max=2.25, n=50),
    'feature': IntegrationConfig(R_max=1, n=50),
}

HIGH = {
    'cloud': IntegrationConfig(R_max=_CUT_OFF, n=500),
    'band1': IntegrationConfig(R_max=_CUT_OFF, n=500),
    'band2': IntegrationConfig(R_max=_CUT_OFF, n=500),
    'band3': IntegrationConfig(R_max=_CUT_OFF, n=500),
    'ring': IntegrationConfig(R_max=2.25, n=200),
    'feature': IntegrationConfig(R_max=1, n=200),
}


INTEGRATION_CONFIGS = {
    'default' : DEFAULT,
    'high' : HIGH,
}