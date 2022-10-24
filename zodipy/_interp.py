from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import astropy.units as u
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from ._constants import SPECIFIC_INTENSITY_UNITS
from ._ipd_model import InterplanetaryDustModel
from ._types import FrequencyOrWavelength


@dataclass
class InterpolatedSourceParameters:
    emissivities: npt.NDArray[np.float64]
    albedos: npt.NDArray[np.float64]
    phase_coefficients: tuple[float, ...]
    solar_irradiance: float


def interpolate_source_parameters(
    model: InterplanetaryDustModel,
    freq: FrequencyOrWavelength,
    weights: npt.NDArray[np.float64],
) -> InterpolatedSourceParameters:

    # Shift spectrum convention between frequency and wavelength (if model units
    # differes from input units) and flip bandpass to make it strictly increasing
    # for np.trapz.
    if not freq.unit.is_equivalent(model.spectrum.unit):
        freq_value = freq.to(model.spectrum.unit, u.spectral()).value
        if weights.size > 1:
            freq_value = np.flip(freq_value)
            weights = np.flip(weights)
            weights = weights / np.trapz(weights, freq_value)
    else:
        freq_value = freq.value

    interpolator = partial(interp1d, x=model.spectrum.value, fill_value="extrapolate")
    emissivities = np.asarray(
        [
            interpolator(y=model.emissivities[comp_label])(freq_value)
            for comp_label in model.comps.keys()
        ]
    )

    if model.albedos is not None:
        albedos = np.asarray(
            [
                interpolator(y=model.albedos[comp_label])(freq_value)
                for comp_label in model.comps.keys()
            ]
        )
    else:
        albedos = np.zeros_like(emissivities)

    if model.phase_coefficients is not None:
        phase_coefficients = interpolator(y=np.asarray(model.phase_coefficients))(
            freq_value
        )

    else:
        phase_coefficients = np.repeat(
            np.zeros((3, 1)), repeats=freq_value.size, axis=-1
        )

    if model.solar_irradiance is not None:
        solar_irradiance = interpolator(y=model.solar_irradiance.value)(freq_value)
        solar_irradiance = (
            u.Quantity(solar_irradiance, model.solar_irradiance.unit)
            .to(SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral())
            .value
        )
    else:
        solar_irradiance = 0

    if freq_value.size > 1:
        bandpass_integrate = lambda quantity: partial(np.trapz, x=freq_value, axis=-1)(
            quantity * weights
        )

        emissivities = bandpass_integrate(emissivities)
        albedos = bandpass_integrate(albedos)
        phase_coefficients = bandpass_integrate(phase_coefficients)
        solar_irradiance = bandpass_integrate(solar_irradiance)

    return InterpolatedSourceParameters(
        emissivities=emissivities,
        albedos=albedos,
        phase_coefficients=tuple(phase_coefficients),
        solar_irradiance=solar_irradiance,
    )
