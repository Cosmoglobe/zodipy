from typing import Callable, Tuple, Dict

import numpy as np


_ZERO = np.finfo(float).eps


class IntegrationConfig:
    """Configuration for the integration strategies in the simulations."""

    def __init__(
        self, 
        R_max: float, 
        n:int, 
        integrator: Callable[[float, float, int], np.ndarray] = np.trapz
    ):
        self.R = np.expand_dims(np.linspace(_ZERO, R_max, n), axis=1)   
        self.dR = np.diff(self.R)
        self.integrator = integrator


class IntegrationConfigFactory:
    """Factory responsible for registring and book-keeping integration configs."""

    def __init__(self) -> None:
        self._configs = {}

    def register_config(
        self, 
        name: str, 
        components: Dict[str, Tuple[float,int]]
    ) -> None:
        """Initializes and stores an integration config."""

        config = {}
        for key, value in components.items():
            config[key] = IntegrationConfig(*value)

        self._configs[name] = config

    def get_config(self, name: str) -> IntegrationConfig: 
        """Returns a registered config."""
        
        config = self._configs.get(name)
        if config is None:
            raise ValueError(
                f'Config {name} is not registered. Available configs are '
                f'{list(self._configs.keys())}'
            )

        return config