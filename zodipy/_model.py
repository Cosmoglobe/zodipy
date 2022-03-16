from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence

from astropy.units import Quantity
import astropy.units as u

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS

__all__ = ("InterplanetaryDustModel", "ModelRegistry")


@dataclass
class InterplanetaryDustModel:
    name: str
    comp_params: dict[CompLabel, dict[str, Any]]
    emissivities: dict[CompLabel, Sequence[float]]
    emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m]
    T_0: Quantity[u.K]
    delta: float
    albedos: Optional[dict[CompLabel, Sequence[float]]] = None
    albedo_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]] = None
    phase_coeffs: Optional[dict[str, Quantity]] = None
    phase_coeffs_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]] = None
    meta: Optional[dict[str, Any]] = field(default_factory=dict)
    comps: dict[CompLabel, Component] = field(init=False)

    def __post_init__(self):
        # Initialize the component classes from the `comp_params` dict.
        self.comps: dict[CompLabel, Component] = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.comp_params.items()
        }

    @property
    def ncomps(self) -> int:
        return len(self.comp_params)


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        comp_params: dict[CompLabel, dict[str, Any]],
        emissivities: dict[CompLabel, Sequence[float]],
        emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m],
        T_0: Quantity[u.K],
        delta: float,
        albedos: Optional[dict[CompLabel, Sequence[float]]] = None,
        albedo_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]] = None,
        phase_coeffs: Optional[dict[str, Quantity]] = None,
        phase_coeffs_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:

        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = InterplanetaryDustModel(
            name=name,
            comp_params=comp_params,
            albedos=albedos,
            albedo_spectrum=albedo_spectrum,
            emissivities=emissivities,
            emissivity_spectrum=emissivity_spectrum,
            phase_coeffs=phase_coeffs,
            phase_coeffs_spectrum=phase_coeffs_spectrum,
            T_0=T_0,
            delta=delta,
            meta=meta,
        )

    def register_model_from_dict(self, model_dict: dict[str, Any]) -> None:
        if (name := model_dict["name"]) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = InterplanetaryDustModel.from_dict(model_dict)

    def get_model(self, name: str) -> InterplanetaryDustModel:
        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self._registry)}"
            )

        return self._registry[name]

    def get_registered_model_names(self) -> list[str]:
        return list(self._registry.keys())


model_registry = ModelRegistry()
