from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import astropy.constants as const
import astropy.units as u
from astropy.units import Quantity
from astropy.utils.data import download_file
import numpy as np
from scipy.interpolate import interp1d

from .source_params import SPECTRUM_DIRBE

# Temporary storage location for Solar irradiance tables
DOWNLOAD_URL = "http://tsih3.uio.no/www_cmb/metins/"

# Tabulated values from the DIRBE IDL Zodiacal emission software. The IDL code
# can be found here: https://lambda.gsfc.nasa.gov/product/cobe/dirbe_zodi_sw.html
DIRBE_SOLAR_IRRADIANCE = [
    2.3405606e8,
    1.2309874e8,
    64292872,
    35733824,
    5763843.0,
    1327989.4,
    230553.73,
    82999.336,
    42346.605,
    14409.608,
] * (u.MJy / u.sr)

SpecificIntensityUnits = u.W / u.Hz / u.m**2 / u.sr


@dataclass(frozen=True)
class SolarIrradianceModel:
    name: str
    spectrum: Quantity[u.Hz] | Quantity[u.m]
    irradiance: Quantity[SpecificIntensityUnits] | Quantity[u.MJy / u.sr]

    def get_solar_irradiance(
        self, frequency: Quantity[u.Hz] | Quantity[u.m], extrapolate: bool
    ) -> float:
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
            SpecificIntensityUnits, equivalencies=u.spectral()
        )

        return solar_flux_specific_intensity_units.value


def dirbe() -> SolarIrradianceModel:
    return SolarIrradianceModel("dirbe", SPECTRUM_DIRBE, DIRBE_SOLAR_IRRADIANCE)


def thuillier2004() -> SolarIrradianceModel:
    TABLE_NAME = "thuillier2004_flux.txt"
    solar_irradiance_table = download_file(DOWNLOAD_URL + TABLE_NAME, cache=True)
    spectrum, irradiance = np.loadtxt(solar_irradiance_table, skiprows=2, unpack=True)

    spectrum *= u.nm
    irradiance *= u.Unit("mW/(m^2 nm sr)")
    irradiance *= spectrum**2 / const.c

    return SolarIrradianceModel("thuillier2004", spectrum, irradiance)


def gueymard2003() -> SolarIrradianceModel:
    TABLE_NAME = "gueymard2003_flux.txt"
    solar_irradiance_table = download_file(DOWNLOAD_URL + TABLE_NAME, cache=True)
    spectrum, irradiance = np.loadtxt(solar_irradiance_table, skiprows=9, unpack=True)

    spectrum *= u.nm
    irradiance *= u.Unit("W/(m^2 nm sr)")
    irradiance *= spectrum**2 / const.c

    return SolarIrradianceModel("gueymard2003", spectrum, irradiance)


SOLAR_FLUX_MODELS: dict[str, Callable[[], SolarIrradianceModel]] = {
    "dirbe": dirbe,
    "thuillier": thuillier2004,
    "gueymard": gueymard2003,
}


def get_solar_irradiance_model(model: str) -> SolarIrradianceModel:
    try:
        return SOLAR_FLUX_MODELS[model]()
    except KeyError:
        raise ValueError(
            f"{model!r} is not a registered Solar irradiance model. Available "
            f"models are: {', '.join(SOLAR_FLUX_MODELS)}."
        )
