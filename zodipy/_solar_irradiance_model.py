from __future__ import annotations

from dataclasses import dataclass, asdict, field

import astropy.constants as const
import astropy.units as u
from astropy.units import Quantity
from astropy.utils.data import download_file
import numpy as np
from scipy.interpolate import interp1d

from ._source_functions import SPECIFIC_INTENSITY_UNITS


@dataclass(frozen=True)
class SolarIrradianceModelFromURL:
    """Model that is downloaded and cached first time it is requested."""

    url: str
    spectrum_unit: u.Unit
    irradiance_unit: u.Unit
    skip_rows: int


@dataclass(frozen=True)
class SolarIrradianceModel:
    """Solar irradiance spectrum given some model."""

    name: str
    spectrum: Quantity[u.Hz] | Quantity[u.m]
    irradiance: Quantity[SPECIFIC_INTENSITY_UNITS] | Quantity[u.MJy / u.sr]

    @classmethod
    def from_url(
        cls,
        name: str,
        url: str,
        spectrum_unit: u.Unit,
        irradiance_unit: u.Unit,
        skip_rows: int = 0,
    ) -> SolarIrradianceModel:
        """
        Initialize a Solar irradiance model from a table on the web. The model
        is downloaded and cached the first time it is requested.
        """

        solar_irradiance_table = download_file(url, cache=True)

        spectrum, irradiance = np.loadtxt(
            solar_irradiance_table, skiprows=skip_rows, unpack=True
        )
        spectrum *= spectrum_unit
        irradiance *= irradiance_unit

        try:
            irradiance = irradiance.to(SPECIFIC_INTENSITY_UNITS)
        except u.UnitConversionError:
            # The irradiance is stored in units of wavelength so we convert to
            # frequency by multiplying by lambda^2/c.
            irradiance = (spectrum**2 / const.c * irradiance).to(
                SPECIFIC_INTENSITY_UNITS
            )

        return SolarIrradianceModel(name, spectrum, irradiance)

    def get_solar_irradiance(
        self, frequency: Quantity[u.Hz] | Quantity[u.m], extrapolate: bool
    ) -> float:
        """Returns the interpolated / extrapolated solar irradiance."""

        frequency = frequency.to(self.spectrum.unit, equivalencies=u.spectral())
        interpolator = interp1d(
            x=self.spectrum.value,
            y=self.irradiance.value,
            fill_value="extrapolate" if extrapolate else None,
        )
        try:
            solar_flux = interpolator(frequency.value)
        except ValueError:
            raise ValueError(
                f"Solar flux model {self.name!r} is only valid in the "
                f"[{self.spectrum.min().value}, {self.spectrum.max().value}] "
                f"{self.spectrum.unit} range."
            )

        solar_flux *= self.irradiance.unit
        solar_flux_specific_intensity_units = solar_flux.to(
            SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral()
        )

        return solar_flux_specific_intensity_units.value


@dataclass
class SolarIrradianceModelRegistry:
    """Container for registered Solar irradiance models."""

    _registry: dict[str, SolarIrradianceModel | SolarIrradianceModelFromURL] = field(
        default_factory=dict
    )

    def register_model_from_url(
        self,
        name: str,
        url: str,
        spectrum_unit: u.Unit,
        irradiance_unit: u.Unit,
        skip_rows: int = 0,
    ) -> None:
        """Registers a model whos spectrum is tabulated on the web."""

        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = SolarIrradianceModelFromURL(
            url, spectrum_unit, irradiance_unit, skip_rows
        )

    def register_model_from_table(
        self,
        name: str,
        spectrum: Quantity[u.Hz] | Quantity[u.m],
        irradiance: Quantity[SPECIFIC_INTENSITY_UNITS] | Quantity[u.MJy / u.sr],
    ) -> None:
        """Registers a model from a spectra."""

        if (name := name.lower()) in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = SolarIrradianceModel(name, spectrum, irradiance)

    def get_model(self, name: str) -> SolarIrradianceModel:
        """Returns a registered Solar irradiance model given a name."""

        try:
            model = self._registry[name]
        except KeyError:
            raise ValueError(
                f"{name!r} is not a registered Solar irradiance model. "
                f"Avaliable models are: {', '.join(self._registry)}."
            )

        if isinstance(model, SolarIrradianceModelFromURL):
            return SolarIrradianceModel.from_url(name, **asdict(model))

        return model


solar_irradiance_model_registry = SolarIrradianceModelRegistry()
