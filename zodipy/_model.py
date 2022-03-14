from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from astropy.units import Quantity
import astropy.units as u

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS

__all__ = ("InterplanetaryDustModel", "ModelRegistry")

ModelDict = Dict[str, Any]


@dataclass
class InterplanetaryDustModel:
    name: str
    comp_params: Dict[CompLabel, Dict[str, Any]]
    emissivities: Dict[CompLabel, List[float]]
    emissivity_spectrum: Union[Quantity[u.Hz], Quantity[u.m]]
    T_0: Quantity[u.K]
    delta: float
    albedos: Optional[Dict[CompLabel, List[float]]] = None
    albedo_spectrum: Optional[Union[Quantity[u.Hz], Quantity[u.m]]] = None
    phase_coeffs: Optional[Dict[str, Quantity]] = None
    phase_coeffs_spectrum: Optional[Union[Quantity[u.Hz], Quantity[u.m]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    comps: Dict[CompLabel, Component] = field(init=False)

    def __post_init__(self):
        self.comps: Dict[CompLabel, Component] = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.comp_params.items()
        }

    @classmethod
    def from_dict(cls, model_dict: ModelDict) -> InterplanetaryDustModel:
        return InterplanetaryDustModel(**model_dict)

    def to_dict(self) -> ModelDict:
        return asdict(self)

    @property
    def ncomps(self) -> int:
        return len(self.comp_params)


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        comp_params: Dict[CompLabel, Dict[str, Any]],
        emissivities: Dict[CompLabel, List[float]],
        emissivity_spectrum: Union[Quantity[u.Hz], Quantity[u.m]],
        T_0: Quantity[u.K],
        delta: float,
        albedos: Optional[Dict[CompLabel, List[float]]] = None,
        albedo_spectrum: Optional[Union[Quantity[u.Hz], Quantity[u.m]]] = None,
        phase_coeffs: Optional[Dict[str, Quantity]] = None,
        phase_coeffs_spectrum: Optional[Union[Quantity[u.Hz], Quantity[u.m]]] = None,
        meta: Optional[Dict[str, Any]] = None,
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

    def register_model_from_dict(self, model_dict: ModelDict) -> None:
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

    def get_registered_model_names(self) -> List[str]:
        return list(self._registry.keys())


model_registry = ModelRegistry()
