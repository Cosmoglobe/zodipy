from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import astropy.units as u
from astropy.units import Quantity

from zodipy._labels import CompLabel


EPS = np.finfo(float).eps * u.AU  # Smallest non zero value.
RADIAL_CUTOFF = 5.2 * u.AU  # Distance to Jupiter in AU.
RING_CUTOFF = 2.25 * u.AU
FEATURE_CUTOFF = 1 * u.AU


@dataclass
class LOSConfigRegistry:
    """Container for registered integration configs.

    By integration config, we mean a mapping of discrete points along a
    line-of-sight per component.
    """

    _registry: dict[str, dict[CompLabel, Quantity[u.AU]]] = field(default_factory=dict)

    def register_config(
        self,
        name: str,
        comps: dict[CompLabel, Quantity],
    ) -> None:
        """Adds a new integration config to the registry."""

        self._registry[name] = comps

    def get_config(self, name: str = "default") -> dict[CompLabel, Quantity[u.AU]]:
        """Returns an integration config from the registry."""

        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a config in the registry. Avaliable configs are "
                f"{', '.join(self._registry)}"
            )
        return self._registry[name]


integration_config_registry = LOSConfigRegistry()

integration_config_registry.register_config(
    name="default",
    comps={
        CompLabel.CLOUD: np.linspace(EPS, RADIAL_CUTOFF, 155),
        CompLabel.BAND1: np.linspace(EPS, RADIAL_CUTOFF, 50),
        CompLabel.BAND2: np.linspace(EPS, RADIAL_CUTOFF, 50),
        CompLabel.BAND3: np.linspace(EPS, RADIAL_CUTOFF, 50),
        CompLabel.RING: np.linspace(EPS, RING_CUTOFF, 50),
        CompLabel.FEATURE: np.linspace(EPS, FEATURE_CUTOFF, 50),
    },
)

DEFAULT_LOS_CONFIG = integration_config_registry.get_config()