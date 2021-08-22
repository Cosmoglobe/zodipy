from typing import Callable

import numpy as np

ZERO = np.finfo(float).eps
RADIAL_CUTOFF = 6


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

    def __init__(
        self, 
        R_max: float, 
        n:int, 
        integrator: Callable[[float, float, int], np.ndarray] = np.trapz
    ):
        self.R = np.expand_dims(np.linspace(ZERO, R_max, n), axis=1)   
        self.dR = np.diff(self.R)
        self.integrator = integrator


DEFAULT = {
    'cloud': IntegrationConfig(RADIAL_CUTOFF, 250),
    'band1': IntegrationConfig(RADIAL_CUTOFF, 50),
    'band2': IntegrationConfig(RADIAL_CUTOFF, 50),
    'band3': IntegrationConfig(RADIAL_CUTOFF, 50),
    'ring': IntegrationConfig(2.25, 50),
    'feature': IntegrationConfig(1, 50),
}

HIGH = {
    'cloud': IntegrationConfig(RADIAL_CUTOFF, 500),
    'band1': IntegrationConfig(RADIAL_CUTOFF, 500),
    'band2': IntegrationConfig(RADIAL_CUTOFF, 500),
    'band3': IntegrationConfig(RADIAL_CUTOFF, 500),
    'ring': IntegrationConfig(2.25, 200),
    'feature': IntegrationConfig(1, 200),
}


INTEGRATION_CONFIGS = {
    'default' : DEFAULT,
    'high' : HIGH,
}