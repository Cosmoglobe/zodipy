from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from ._ipd_model import InterplanetaryDustModel
from ._source_funcs import SPECIFIC_INTENSITY_UNITS
from ._typing import FrequencyOrWavelength


@dataclass
class InterpolatedSourceParameters:
    emissivities: NDArray[np.floating]
    albedos: NDArray[np.floating]
    phase_coefficients: tuple[float, ...]
    solar_irradiance: float


def interpolate_source_parameters(
    model: InterplanetaryDustModel,
    freq: FrequencyOrWavelength,
    weights: NDArray[np.floating] | None = None,
) -> InterpolatedSourceParameters:

    interpolator = partial(interp1d, x=model.spectrum, fill_value="extrapolate")
    emissivities = np.asarray(
        [
            interpolator(y=model.emissivities[comp_label])(freq)
            for comp_label in model.comps.keys()
        ]
    )
    if model.albedos is not None:
        albedos = np.asarray(
            [
                interpolator(y=model.albedos[comp_label])(freq)
                for comp_label in model.comps.keys()
            ]
        )
    else:
        albedos = np.zeros_like(emissivities)

    if model.phase_coefficients is not None:  # double check this
        phase_coefficients = interpolator(y=np.asarray(model.phase_coefficients))(freq)

    else:

        phase_coefficients = np.repeat(np.zeros((3, 1)), repeats=freq.size, axis=-1)

    if model.solar_irradiance is not None:
        solar_irradiance = interpolator(y=model.solar_irradiance)(freq)
        solar_irradiance = (
            (solar_irradiance * model.solar_irradiance.unit)
            .to(SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral())
            .value
        )
    else:
        solar_irradiance = 0

    if weights is not None:
        emissivities = np.trapz(weights * emissivities, freq.value, axis=-1)
        albedos = np.trapz(weights * albedos, freq.value, axis=-1)
        phase_coefficients = np.trapz(weights * phase_coefficients, freq.value, axis=-1)
        solar_irradiance = np.trapz(weights * solar_irradiance, freq.value, axis=-1)

    return InterpolatedSourceParameters(
        emissivities=emissivities,
        albedos=albedos,
        phase_coefficients=tuple(phase_coefficients),
        solar_irradiance=solar_irradiance,
    )
