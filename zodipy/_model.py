from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from astropy.units import Quantity
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS

__all__ = ("InterplanetaryDustModel", "ModelRegistry")


class CloudComponentMissingError(Exception):
    """Raised if an `InterplanetaryDustModel` model is missing the Diffuse Cloud."""

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
    meta: dict[str, Any] = field(default_factory=dict)
    comps: dict[CompLabel, Component] = field(init=False)

    def __post_init__(self) -> None:
        if CompLabel.CLOUD not in self.comp_params:
            raise CloudComponentMissingError(
                f"""InterplanetaryDustModel {self.name!r} is missing the "
                "Diffuse Cloud component."""
            )
        self.comps: dict[CompLabel, Component] = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.comp_params.items()
        }

    @property
    def ncomps(self) -> int:
        return len(self.comp_params)

    @property
    def cloud_offset(self) -> NDArray[np.floating]:
        return self.comps[CompLabel.CLOUD].X_0


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

        if meta is None:
            meta = {}

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
