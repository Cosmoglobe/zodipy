from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from astropy.units import Quantity
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS


@dataclass
class Model:
    name: str
    comp_params: dict[CompLabel, dict[str, Any]]
    emissivities: dict[CompLabel, tuple[float, ...]]
    emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m]
    T_0: Quantity[u.K]
    delta: float
    albedos: dict[CompLabel, tuple[float, ...]] | None = None
    albedo_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None
    phase_coeffs: dict[str, Quantity] | None = None
    phase_coeffs_spectrum: Quantity[u.Hz] | Quantity[u.m] | None = None
    comps: dict[CompLabel, Component] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.comps = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.comp_params.items()
        }

    @property
    def ncomps(self) -> int:
        return len(self.comps)

    @property
    def cloud_offset(self) -> NDArray[np.floating]:
        return self.comps[CompLabel.CLOUD].X_0

    def get_interp_spectral_params(
        self, freq: Quantity[u.GHz] | Quantity[u.m]
    ) -> dict[CompLabel, dict[str, float]]:
        """Returns interpolated/extrapolated spectral parameters given a frequency."""

        interp_params: dict[CompLabel, dict[str, float]] = {
            comp_label: {} for comp_label in self.emissivities
        }

        for comp, emiss_Sequence in self.emissivities.items():
            interp_params[comp]["emissivity"] = np.interp(
                freq.to(self.emissivity_spectrum.unit, equivalencies=u.spectral()),
                self.emissivity_spectrum,
                emiss_Sequence,
            )

        if self.albedos is not None and self.albedo_spectrum is not None:
            for comp, albedo_Sequence in self.albedos.items():
                interp_params[comp]["albedo"] = np.interp(
                    freq.to(self.albedo_spectrum.unit, equivalencies=u.spectral()),
                    self.albedo_spectrum,
                    albedo_Sequence,
                )
        else:
            for comp in self.emissivities:
                interp_params[comp]["albedo"] = 0.0

        return interp_params

    def get_interp_phase_coeffs(
        self, freq: Quantity[u.GHz] | Quantity[u.m]
    ) -> list[float]:
        """Returns interpolated/extrapolated phase coefficients given a frequency."""

        if self.phase_coeffs is not None and self.phase_coeffs_spectrum is not None:
            interp_phase_coeffs = [
                np.interp(
                    freq.to(
                        self.phase_coeffs_spectrum.unit, equivalencies=u.spectral()
                    ),
                    self.phase_coeffs_spectrum,
                    coeff,
                ).value
                for coeff in self.phase_coeffs.values()
            ]
        else:
            interp_phase_coeffs = [0.0 for _ in range(3)]

        return interp_phase_coeffs


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: dict[str, Model] = field(init=False, default_factory=dict)

    def register_model(
        self,
        name: str,
        comp_params: dict[CompLabel, dict[str, Any]],
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
            comp_params=comp_params,
            albedos=albedos,
            albedo_spectrum=albedo_spectrum,
            emissivities=emissivities,
            emissivity_spectrum=emissivity_spectrum,
            phase_coeffs=phase_coeffs,
            phase_coeffs_spectrum=phase_coeffs_spectrum,
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
