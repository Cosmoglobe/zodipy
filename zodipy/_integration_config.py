from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from zodipy._labels import Label


EPS = np.finfo(float).eps * u.AU  # Smallest non zero value.
RADIAL_CUTOFF = 5.2 * u.AU  # Distance to Jupiter in AU.
RING_CUTOFF = 2.25 * u.AU
FEATURE_CUTOFF = 1 * u.AU


@dataclass
class IntegrationConfigRegistry:
    """Container for registered integration configs.

    By integration config, we mean a mapping of discrete points along a
    line-of-sight per component.
    """

    _registry: Dict[str, Dict[Label, Quantity[u.AU]]] = field(default_factory=dict)

    def register_config(
        self,
        name: str,
        components: Dict[Label, Quantity],
    ) -> None:
        """Adds a new integration config to the registry."""

        self._registry[name] = components

    def get_config(self, name: str = "default") -> Dict[Label, Quantity]:
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
        Label.RING: np.linspace(EPS, RING_CUTOFF, 50),
        Label.FEATURE: np.linspace(EPS, FEATURE_CUTOFF, 50),
    },
)

DEFAULT_INTEGRATION_CONFIG = integration_config_registry.get_config()