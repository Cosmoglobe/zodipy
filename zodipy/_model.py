from __future__ import annotations

from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from ._component import Component
from ._component_label import ComponentLabel

from._solar_irradiance_model import SolarIrradianceModel
from .solar_irradiance_models import solar_irradiance_model_registry
from .source_parameters import DELTA_DIRBE, T_0_DIRBE


@dataclass
class Model:
    """Interplanetary Dust model.

    This dataclass acts as a container for the various parameters in a
    Interplanetary Dust model and provides a method to interpolate / extrapolate
    the spectral model parameters to other frequencies or wavelengths.
    """

    name: str
    comps: dict[ComponentLabel, Component]
    spectrum: u.Quantity[u.Hz] | u.Quantity[u.m]
    emissivities: dict[ComponentLabel, tuple[float, ...]]
    albedos: dict[ComponentLabel, tuple[float, ...]] | None = None
    solar_irradiance_model: SolarIrradianceModel | None = None
    phase_coefficients: list[tuple[float, ...]] | None = None
    T_0: float = T_0_DIRBE
    delta: float = DELTA_DIRBE

    @property
    def n_comps(self) -> int:
        return len(self.comps)

    def interpolate_source_parameters(
        self, comp_label: ComponentLabel, freq: u.Quantity[u.GHz] | u.Quantity[u.m]
    ) -> tuple[float, float, tuple[float, float, float]]:
        """
        Returns interpolated/extrapolated source parameters for a component
        given a frequency.
        """

        freq = freq.to(self.spectrum.unit, equivalencies=u.spectral())
        emissivity_interpolator = interp1d(
            x=self.spectrum,
            y=self.emissivities[comp_label],
            fill_value="extrapolate",
        )
        emissivity = emissivity_interpolator(freq)

        if self.albedos is not None:
            albedo_interpolator = interp1d(
                x=self.spectrum,
                y=self.albedos[comp_label],
                fill_value="extrapolate",
            )
            albedo = albedo_interpolator(freq)
        else:
            albedo = 0.0

        if self.phase_coefficients is not None:
            phase_coefficient_interpolator = interp1d(
                x=self.spectrum,
                y=np.asarray(self.phase_coefficients),
                fill_value="extrapolate",
            )
            phase_coefficient = phase_coefficient_interpolator(freq)
        else:
            phase_coefficient = [0.0 for _ in range(3)]

        return emissivity, albedo, tuple(phase_coefficient)

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
        comps: dict[ComponentLabel, Component],
        spectrum: u.Quantity[u.Hz] | u.Quantity[u.m],
        emissivities: dict[ComponentLabel, tuple[float, ...]],
        albedos: dict[ComponentLabel, tuple[float, ...]] | None = None,
        solar_irradiance_model: str | None = None,
        phase_coefficients: list[tuple[float, ...]] | None = None,
        T_0: float = T_0_DIRBE,
        delta: float = DELTA_DIRBE,
    ) -> None:
        """Registers a new model.

        For an example of how to register a custom model, see the documentation
        on https://zodipy.readthedocs.io/en/latest/ (coming soon).

        Parameters
        ----------
        name
            String representing the name of the model. This is the name that is
            used for the 'model' argument when initializing `Zodipy`.
        comps
            Dict mapping `CompLabel`s to `Component` classes.
        spectrum
            The spectrum (frequency or length units) corresponding to the
            frequencies used to estimate the spectral parameters.
        emissivities
            Emissivity factor fits for each IPD component at the frequencies
            corresponding to 'spectrum'.
        albedos
            Albedo factor fits for each IPD component at the frequencies
            corresponding to 'spectrum'.
        solar_irradiance_model
            Model for the solar irradiance.
        phase_coefficients
            Coefficient fits for the phase function at the frequencies
            corresponding to 'spectrum'.
        T_0
            Dust grain temperature at 1 AU. Defaults to the DIRBE model value.
        delta
            Dust grain temperature powerlaw parameter describing how the
            temperature falls with radial distance from the Sun. Defaults to the
            DIRBE model value.
        """

        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        if solar_irradiance_model is not None:
            solar_model = solar_irradiance_model_registry.get_model(
                solar_irradiance_model
            )
        else:
            solar_model = solar_irradiance_model

        if albedos is not None and solar_irradiance_model is None:
            raise ValueError(
                "albedos are provided but no solar irradiance model is provided."
            )
        self._registry[name] = Model(
            name=name,
            comps=comps,
            spectrum=spectrum,
            emissivities=emissivities,
            T_0=T_0,
            delta=delta,
            albedos=albedos,
            solar_irradiance_model=solar_model,
            phase_coefficients=phase_coefficients,
        )

    def get_model(self, name: str) -> Model:
        """Returns a registered model given a name."""

        if (name := name.lower()) not in self._registry:
            raise ValueError(
                f"{name!r} is not a registered Interplanetary Dust model. "
                f"Avaliable models are: {', '.join(self._registry)}."
            )

        return self._registry[name]


model_registry = ModelRegistry()
