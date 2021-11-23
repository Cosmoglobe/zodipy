from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from zodipy._component_labels import ComponentLabel


EPS = np.finfo(float).eps  # Smallest non zero value.
RADIAL_CUTOFF = 6  # Distance to Jupiter in AU.


@dataclass
class IntegrationConfigRegistry:
    registry: Dict[str, Dict[ComponentLabel, np.ndarray]] = field(default_factory=dict)

    def register_config(
        self,
        name: str,
        components: Dict[ComponentLabel, np.ndarray],
    ) -> None:

        self.registry[name] = components

    def get_config(self, name: str) -> Dict[ComponentLabel, np.ndarray]:

        if name not in self.registry:
            raise ModuleNotFoundError(
                f"{name} is not a config in the registry. Avaliable configs are "
                f"{', '.join(self.registry)}"
            )
        return self.registry[name]


integration_config_registry = IntegrationConfigRegistry()

integration_config_registry.register_config(
    name="default",
    components={
        ComponentLabel.CLOUD: np.linspace(EPS, RADIAL_CUTOFF, 125),
        ComponentLabel.BAND1: np.linspace(EPS, RADIAL_CUTOFF, 50),
        ComponentLabel.BAND2: np.linspace(EPS, RADIAL_CUTOFF, 50),
        ComponentLabel.BAND3: np.linspace(EPS, RADIAL_CUTOFF, 50),
        ComponentLabel.RING: np.linspace(EPS, 2.25, 50),
        ComponentLabel.FEATURE: np.linspace(EPS, 1, 50),
    },
)

DEFAULT_INTEGRATION_CONFIG = integration_config_registry.get_config("default")
