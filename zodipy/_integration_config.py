from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from zodipy._labels import Label


EPS = np.finfo(float).eps  # Smallest non zero value.
RADIAL_CUTOFF = 6  # Distance to Jupiter in AU.


@dataclass
class IntegrationConfigRegistry:
    """Container for registered integration configs.
    
    By integration config, we mean a mapping of discrete points along a 
    line-of-sight per component.
    """

    _registry: Dict[str, Dict[Label, np.ndarray]] = field(default_factory=dict)

    def register_config(
        self,
        name: str,
        components: Dict[Label, np.ndarray],
    ) -> None:
        """Adds a new integration config to the registry."""

        self._registry[name] = components

    def get_config(self, name: str) -> Dict[Label, np.ndarray]:
        """Returns an integration config from the registry."""

        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a config in the registry. Avaliable configs are "
                f"{', '.join(self._registry)}"
            )
        return self._registry[name]


integration_config_registry = IntegrationConfigRegistry()

integration_config_registry.register_config(
    name="default",
    components={
        Label.CLOUD: np.linspace(EPS, RADIAL_CUTOFF, 155),
        Label.BAND1: np.linspace(EPS, RADIAL_CUTOFF, 50),
        Label.BAND2: np.linspace(EPS, RADIAL_CUTOFF, 50),
        Label.BAND3: np.linspace(EPS, RADIAL_CUTOFF, 50),
        Label.RING: np.linspace(EPS, 2.25, 50),
        Label.FEATURE: np.linspace(EPS, 1, 50),
    },
)

DEFAULT_INTEGRATION_CONFIG = integration_config_registry.get_config("default")

