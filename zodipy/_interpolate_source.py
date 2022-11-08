from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, TypeVar

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from zodipy._bandpass import Bandpass
from zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall

InterplanetaryDustModelT = TypeVar(
    "InterplanetaryDustModelT", bound=InterplanetaryDustModel
)

"""Returns the source parameters for a given bandpass and model. Must match arguments in the emission fns."""
SourceParametersFn = Callable[[Bandpass, InterplanetaryDustModelT], Dict[str, Any]]


def get_source_parameters_kelsall(bandpass: Bandpass, model: Kelsall) -> dict[str, Any]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    interpolator = partial(interp1d, x=model.spectrum.value, fill_value="extrapolate")
    emissivities = np.asarray(
        [
            interpolator(y=model.emissivities[comp_label])(bandpass.frequencies.value)
            for comp_label in model.comps.keys()
        ]
    )

    if model.albedos is not None:
        albedos = np.asarray(
            [
                interpolator(y=model.albedos[comp_label])(bandpass.frequencies.value)
                for comp_label in model.comps.keys()
            ]
        )
    else:
        albedos = np.zeros_like(emissivities)

    if model.phase_coefficients is not None:
        phase_coefficients = interpolator(y=np.asarray(model.phase_coefficients))(
            bandpass.frequencies.value
        )

    else:
        phase_coefficients = np.repeat(
            np.zeros((3, 1)), repeats=bandpass.frequencies.size, axis=-1
        )

    if model.solar_irradiance is not None:
        solar_irradiance = interpolator(y=model.solar_irradiance.value)(
            bandpass.frequencies.value
        )
        solar_irradiance = u.Quantity(
            solar_irradiance, model.solar_irradiance.unit
        ).to_value(SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral())
    else:
        solar_irradiance = 0

    if bandpass.frequencies.size > 1:
        emissivities = bandpass.integrate(emissivities)
        albedos = bandpass.integrate(albedos)
        phase_coefficients = bandpass.integrate(phase_coefficients)
        solar_irradiance = bandpass.integrate(solar_irradiance)

    return {
        "emissivities": emissivities,
        "albedos": albedos,
        "phase_coefficients": tuple(phase_coefficients),
        "solar_irradiance": solar_irradiance,
        "T_0": model.T_0,
        "delta": model.delta,
    }


def get_source_parameters_rmm(bandpass: Bandpass, model: RRM) -> dict[str, Any]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    calibration = u.Quantity(model.calibration, u.MJy / u.AU).to_value(u.Jy / u.cm)
    calibration = interp1d(
        x=model.spectrum.value, y=calibration, fill_value="extrapolate"
    )(bandpass.frequencies.value)

    if bandpass.frequencies.size > 1:
        calibration = bandpass.integrate(calibration)

    return {
        "calibration": calibration,
        "T_0": tuple(model.T_0[comp] for comp in model.comps.keys()),
        "delta": tuple(model.delta[comp] for comp in model.comps.keys()),
    }


SOURCE_PARAMS_MAPPING: dict[type[InterplanetaryDustModel], SourceParametersFn] = {
    Kelsall: get_source_parameters_kelsall,
    RRM: get_source_parameters_rmm,
}
