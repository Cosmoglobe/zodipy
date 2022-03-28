from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from astropy.units import Quantity
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS


@dataclass
class Model:
    name: str
    component_parameters: dict[CompLabel, dict[str, Any]]
    emissivities: dict[CompLabel, tuple[float, ...]]
    emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m]
    T_0: Quantity[u.K]
    delta: float
    albedos: dict[CompLabel, tuple[float, ...]] | None = None
    albedos_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None
    phase_coefficients: dict[str, Quantity] | None = None
    phase_coefficients_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None
    comps: dict[CompLabel, Component] = field(init=False)
    cloud_offset: NDArray[np.floating] = field(init=False)

    def __post_init__(self) -> None:
        self.comps = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.component_parameters.items()
        }
        self.cloud_offset = self.comps[CompLabel.CLOUD].X_0

    @property
    def ncomps(self) -> int:
        return len(self.comps)

    def get_extrapolated_parameters(
        self, freq: Quantity[u.GHz] | Quantity[u.m]
    ) -> dict[CompLabel, tuple[float, Optional[float], Optional[list[float]]]]:
        """Returns interpolated/extrapolated spectral parameters given a frequency."""

        extrapolated_parameters: dict[
            CompLabel, tuple[float, Optional[float], Optional[list[float]]]
        ] = {}

        for comp in self.comps:
            emissivity = np.interp(
                freq.to(self.emissivity_spectrum.unit, equivalencies=u.spectral()),
                self.emissivity_spectrum,
                self.emissivities[comp],
            )

            if self.albedos is not None:
                if self.albedos_spectrum is None:
                    raise ValueError(
                        "cannot extrapolate in albedos without a abledo spectrum"
                    )
                albedo = np.interp(
                    freq.to(self.albedos_spectrum.unit, equivalencies=u.spectral()),
                    self.albedos_spectrum,
                    self.albedos[comp],
                )
            else:
                albedo = None
            if self.phase_coefficients:
                if self.phase_coefficients_spectrum is None:
                    raise ValueError(
                        "cannot extrapolate in phase coefficients without a phase "
                        "coefficient spectrum"
                    )
                phase_coefficient = [
                    np.interp(
                        freq.to(
                            self.phase_coefficients_spectrum.unit,
                            equivalencies=u.spectral(),
                        ),
                        self.phase_coefficients_spectrum,
                        coeff,
                    ).value
                    for coeff in self.phase_coefficients.values()
                ]
            else:
                phase_coefficient = None
            extrapolated_parameters[comp] = emissivity, albedo, phase_coefficient

        return extrapolated_parameters


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: dict[str, Model] = field(init=False, default_factory=dict)

    def register_model(
        self,
        name: str,
        component_parameters: dict[CompLabel, dict[str, Any]],
        emissivities: dict[CompLabel, tuple[float, ...]],
        emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m],
        T_0: Quantity[u.K],
        delta: float,
        albedos: dict[CompLabel, tuple[float, ...]] | None = None,
        albedo_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None,
        phase_coeffs: dict[str, Quantity] | None = None,
        phase_coeffs_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None,
    ) -> None:

        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = Model(
            name=name,
            component_parameters=component_parameters,
            albedos=albedos,
            albedos_spectrum=albedo_spectrum,
            emissivities=emissivities,
            emissivity_spectrum=emissivity_spectrum,
            phase_coefficients=phase_coeffs,
            phase_coefficients_spectrum=phase_coeffs_spectrum,
            T_0=T_0,
            delta=delta,
        )

    def get_model(self, name: str) -> Model:
        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self._registry)}"
            )

        return self._registry[name]

    def get_registered_model_names(self) -> list[str]:
        return list(self._registry.keys())


model_registry = ModelRegistry()
