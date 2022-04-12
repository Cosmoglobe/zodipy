from __future__ import annotations

from dataclasses import dataclass, field

from astropy.units import Quantity
import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from ._component import Component
from ._component_label import ComponentLabel
from .source_parameters import T_0_DIRBE, DELTA_DIRBE


@dataclass(frozen=True)
class Model:
    """Interplanetary Dust model.

    This dataclass acts as a container for the various parameters in a
    Interplanetary Dust model and provides a method to interpolate / extrapolate
    the spectral model parameters to other frequencies or wavelengths.
    """

    name: str
    components: dict[ComponentLabel, Component]
    spectrum: Quantity[u.Hz] | Quantity[u.m]
    emissivities: dict[ComponentLabel, tuple[float, ...]]
    albedos: dict[ComponentLabel, tuple[float, ...]] | None = None
    phase_coefficients: list[tuple[float, ...]] | None = None
    T_0: float = T_0_DIRBE
    delta: float = DELTA_DIRBE

    @property
    def n_components(self) -> int:
        return len(self.components)

    def validate_frequency(
        self, frequency: Quantity[u.GHz] | Quantity[u.m], extrapolate: bool
    ) -> None:
        """
        Raises ValueError if the requested frequency is out of range the range
        covered by the model.
        """

        if extrapolate:
            return

        frequency = frequency.to(self.spectrum.unit, equivalencies=u.spectral())
        spectrum_min, spectrum_max = self.spectrum.min(), self.spectrum.max()
        if not (spectrum_min <= frequency <= spectrum_max):
            raise ValueError(
                f"model {self.name!r} is only valid in the [{spectrum_min.value},"
                f" {spectrum_max.value}] {self.spectrum.unit} range."
            )

    def get_source_parameters(
        self,
        component_label: ComponentLabel,
        frequency: Quantity[u.GHz] | Quantity[u.m],
    ) -> tuple[float, float, tuple[float, float, float]]:
        """
        Returns interpolated/extrapolated source parameters for a component
        given a frequency.
        """

        frequency = frequency.to(self.spectrum.unit, equivalencies=u.spectral())
        emissivity_interpolator = interp1d(
            x=self.spectrum,
            y=self.emissivities[component_label],
            fill_value="extrapolate",
        )
        emissivity = emissivity_interpolator(frequency)

        if self.albedos is not None:
            albedo_interpolator = interp1d(
                x=self.spectrum,
                y=self.albedos[component_label],
                fill_value="extrapolate",
            )
            albedo = albedo_interpolator(frequency)
        else:
            albedo = 0.0

        if self.phase_coefficients is not None:
            phase_coefficient_interpolator = interp1d(
                x=self.spectrum,
                y=np.asarray(self.phase_coefficients),
                fill_value="extrapolate",
            )
            phase_coefficient = phase_coefficient_interpolator(frequency)
        else:
            phase_coefficient = [0.0 for _ in range(3)]

        return emissivity, albedo, tuple(phase_coefficient)

    def __repr__(self) -> str:
        component_names = [component_label.value for component_label in self.components]
        repr = f"{type(self).__name__}( \n"
        repr += f"   name: {self.name!r},\n"
        repr += "   components: (\n"
        for name in component_names:
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
        components: dict[ComponentLabel, Component],
        spectrum: Quantity[u.Hz] | Quantity[u.m],
        emissivities: dict[ComponentLabel, tuple[float, ...]],
        albedos: dict[ComponentLabel, tuple[float, ...]] | None = None,
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
        components
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

        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = Model(
            name=name,
            components=components,
            spectrum=spectrum,
            emissivities=emissivities,
            T_0=T_0,
            delta=delta,
            albedos=albedos,
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
