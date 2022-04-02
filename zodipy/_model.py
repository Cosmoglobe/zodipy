from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

from astropy.units import Quantity
import astropy.units as u
import numpy as np

from zodipy._components import Component
from zodipy._labels import CompLabel, LABEL_TO_CLASS
from zodipy.source_params import T_0_DIRBE, delta_DIRBE


@dataclass
class Model:
    name: str
    component_parameters: dict[CompLabel, dict[str, Any]]
    spectrum: Quantity[u.Hz] | Quantity[u.m]
    emissivities: dict[CompLabel, tuple[float, ...]]
    albedos: dict[CompLabel, tuple[float, ...]] | None = None
    phase_coefficients: list[tuple[float, ...]] | None = None
    T_0: float = T_0_DIRBE
    delta: float = delta_DIRBE
    comps: dict[CompLabel, Component] = field(init=False)

    def __post_init__(self) -> None:
        self.comps = {
            comp: LABEL_TO_CLASS[comp](**params)
            for comp, params in self.component_parameters.items()
        }

    @property
    def ncomps(self) -> int:
        return len(self.comps)

    def get_extrapolated_parameters(
        self, freq: Quantity[u.GHz] | Quantity[u.m]
    ) -> dict[CompLabel, tuple[float, Optional[float], Optional[tuple[float, ...]]]]:
        """Returns interpolated/extrapolated spectral parameters given a frequency."""

        extrapolated_parameters: dict[
            CompLabel, tuple[float, Optional[float], Optional[tuple[float, ...]]]
        ] = {}
        freq = freq.to(self.spectrum.unit, equivalencies=u.spectral())

        for comp in self.comps:
            emissivity = np.interp(freq, self.spectrum, self.emissivities[comp])
            if self.albedos is not None:
                albedo = np.interp(freq, self.spectrum, self.albedos[comp])
            else:
                albedo = None
            if self.phase_coefficients:
                phase_coefficient = tuple(
                    [
                        np.interp(freq, self.spectrum, coeff)
                        for coeff in self.phase_coefficients
                    ]
                )
            else:
                phase_coefficient = None

            extrapolated_parameters[comp] = emissivity, albedo, phase_coefficient

        return extrapolated_parameters


@dataclass
class ModelRegistry:
    """Container for registered models."""

    _registry: dict[str, Model] = field(init=False, repr=False, default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Returns a list of registered model names."""

        return list(self._registry.keys())

    def register_model(
        self,
        name: str,
        component_parameters: dict[CompLabel, dict[str, Any]],
        spectrum: Quantity[u.Hz] | Quantity[u.m],
        emissivities: dict[CompLabel, tuple[float, ...]],
        albedos: dict[CompLabel, tuple[float, ...]] | None = None,
        phase_coefficients: list[tuple[float, ...]] | None = None,
        T_0: float = T_0_DIRBE,
        delta: float = delta_DIRBE,
    ) -> None:
        """Registers a new model.

        For an example of how to register a custom model, see the documentation
        on https://zodipy.readthedocs.io/en/latest/ (coming soon).

        Parameters
        ----------
        name
            String representing the name of the model. This is the name that is
            used for the 'model' argument when initializing `Zodipy`.
        component_parameters
            Dictionary mapping the geometrical model parameters to the
            respective IPD components. The keys must be CompLabel enums.
        spectrum
            The spectrum (frequency or length units) corresponding to the
            frequencies used to estimate the spectral parameters.
        emissivities
            Emissivity factor fits for each IPD component at the frequencies
            corresponding to 'spectrum'.
        albedos
            Albedo factor fits for each IPD component at the frequencies
            corresponding to 'spectrum'.
        phase_coefficients
            Coefficient fits for the phase function at the frequencies
            corresponding to 'spectrum'.
        T_0
            Interplanetary temperature at 1 AU. Defaults to the DIRBE model value.
        delta
            Interplanetary temperatue powerlaw parameter describing how the
            interplanetary temperature falls with radial distance from the Sun.
            Defaults to the DIRBE model value.
        """

        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = Model(
            name=name,
            component_parameters=component_parameters,
            spectrum=spectrum,
            emissivities=emissivities,
            T_0=T_0,
            delta=delta,
            albedos=albedos,
            phase_coefficients=phase_coefficients,
        )

    def get_model(self, name: str) -> Model:
        """Returns a registered model given a name."""

        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self._registry)}"
            )

        return self._registry[name]


model_registry = ModelRegistry()
