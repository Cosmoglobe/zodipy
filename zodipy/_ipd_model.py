from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import astropy.units as u

from ._ipd_comps import Component, ComponentLabel
from .source_params import DELTA_DIRBE, T_0_DIRBE


@dataclass
class InterplanetaryDustModel:
    """Container for interplanetary dust components and model parameters."""

    name: str
    comps: Mapping[ComponentLabel, Component]
    spectrum: u.Quantity[u.Hz] | u.Quantity[u.m]
    emissivities: Mapping[ComponentLabel, Sequence[float]]
    albedos: Mapping[ComponentLabel, Sequence[float]] | None = None
    solar_irradiance: u.Quantity[u.MJy / u.sr] | None = None
    phase_coefficients: Sequence[Sequence[float]] | None = None
    T_0: float = T_0_DIRBE
    delta: float = DELTA_DIRBE

    @property
    def n_comps(self) -> int:
        return len(self.comps)

    def __repr__(self) -> str:
        comp_names = [comp_label.value for comp_label in self.comps]
        repr = f"{type(self).__name__}( \n"
        repr += f"   name: {self.name!r},\n"
        repr += "   components: (\n"
        for name in comp_names:
            repr += f"      {name!r},\n"
        repr += "   ),\n"
        repr += "   thermal: True,\n"
        repr += f"   scattering: {self.albedos is not None},\n"
        repr += ")"

        return repr


@dataclass
class InterplanetaryDustModelRegistry:
    """Container for registered models."""

    _registry: dict[str, InterplanetaryDustModel] = field(
        init=False, repr=False, default_factory=dict
    )

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    def register_model(
        self,
        name: str,
        comps: Mapping[ComponentLabel, Component],
        spectrum: u.Quantity[u.Hz] | u.Quantity[u.m],
        emissivities: Mapping[ComponentLabel, Sequence[float]],
        albedos: Mapping[ComponentLabel, Sequence[float]] | None = None,
        solar_irradiance: u.Quantity[u.MJy / u.sr] | None = None,
        phase_coefficients: Sequence[Sequence[float]] | None = None,
        T_0: float = T_0_DIRBE,
        delta: float = DELTA_DIRBE,
    ) -> None:
        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        if albedos is not None and solar_irradiance is None:
            raise ValueError(
                "albedos are provided but no solar irradiance is provided."
            )
        self._registry[name] = InterplanetaryDustModel(
            name=name,
            comps=comps,
            spectrum=spectrum,
            emissivities=emissivities,
            T_0=T_0,
            delta=delta,
            albedos=albedos,
            solar_irradiance=solar_irradiance,
            phase_coefficients=phase_coefficients,
        )

    def get_model(self, name: str) -> InterplanetaryDustModel:
        if (name := name.lower()) not in self._registry:
            raise ValueError(
                f"{name!r} is not a registered Interplanetary Dust model. "
                f"Avaliable models are: {', '.join(self._registry)}."
            )

        return self._registry[name]


model_registry = InterplanetaryDustModelRegistry()
