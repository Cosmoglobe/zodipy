from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from zodipy._ipd_comps import Component, ComponentLabel
from zodipy._types import FrequencyOrWavelength


@dataclass
class InterplanetaryDustModel(ABC):
    comps: Mapping[ComponentLabel, Component]
    spectrum: FrequencyOrWavelength

    def to_dict(self) -> dict:
        _dict: dict = {}
        for key, value in vars(self).items():
            if key == "comps":
                _dict[key] = {}
                for comp_key, comp_value in value.items():
                    _dict[key][comp_key.value] = {
                        k: v
                        for k, v in vars(comp_value).items()
                        if comp_value.__dataclass_fields__[k].init
                    }
            elif isinstance(value, dict):
                _dict[key] = {k.value: v for k, v in value.items()}
            else:
                _dict[key] = value

        return _dict


@dataclass
class Kelsall(InterplanetaryDustModel):
    T_0: float
    delta: float
    emissivities: Mapping[ComponentLabel, Sequence[float]]
    albedos: Mapping[ComponentLabel, Sequence[float]] | None = None
    solar_irradiance: Sequence[float] | None = None  # In units of MJy/sr
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
