from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import astropy.units as u

from zodipy._ipd_comps import Component, ComponentLabel
from zodipy._types import FrequencyOrWavelength


@dataclass
class InterplanetaryDustModel(ABC):
    comps: Mapping[ComponentLabel, Component]
    spectrum: FrequencyOrWavelength


@dataclass
class Kelsall(InterplanetaryDustModel):
    T_0: float
    delta: float
    emissivities: Mapping[ComponentLabel, Sequence[float]]
    albedos: Mapping[ComponentLabel, Sequence[float]] | None = None
    solar_irradiance: u.Quantity[u.MJy / u.sr] | None = None
    phase_coefficients: Sequence[Sequence[float]] | None = None


@dataclass
class RRM(InterplanetaryDustModel):
    T_0: Mapping[ComponentLabel, float]
    delta: Mapping[ComponentLabel, float]
    calibration: Sequence[float]


@dataclass
class InterplanetaryDustModelRegistry:
    """Container for registered models."""

    _registry: dict[str, InterplanetaryDustModel] = field(
        init=False, default_factory=dict
    )

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    def register_model(
        self,
        name: str,
        model: InterplanetaryDustModel,
    ) -> None:
        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = model

    def get_model(self, name: str) -> InterplanetaryDustModel:
        if (name := name.lower()) not in self._registry:
            raise ValueError(
                f"{name!r} is not a registered Interplanetary Dust model. "
                f"Avaliable models are: {', '.join(self._registry)}."
            )

        return self._registry[name]


model_registry = InterplanetaryDustModelRegistry()
