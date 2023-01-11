from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, TypeVar, Union

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

from zodipy._bandpass import Bandpass
from zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipy._ipd_comps import ComponentLabel
from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall

InterplanetaryDustModelT = TypeVar(
    "InterplanetaryDustModelT", bound=InterplanetaryDustModel
)

"""Returns the source parameters for a given bandpass and model. 
Must match arguments in the emission fns."""
GetSourceParametersFn = Callable[
    [Bandpass, InterplanetaryDustModelT], Dict[Union[ComponentLabel, str], Any]
]


def get_source_parameters_kelsall_comp(
    bandpass: Bandpass, model: Kelsall
) -> dict[ComponentLabel | str, dict[str, Any]]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    spectrum = (
        model.spectrum.to_value(u.Hz)
        if model.spectrum.unit.is_equivalent(u.Hz)
        else model.spectrum.to_value(u.micron)
    )

    interpolator = partial(interp1d, x=spectrum, fill_value="extrapolate")

    source_parameters: dict[ComponentLabel | str, dict[str, Any]] = {}
    for comp_label in model.comps.keys():
        source_parameters[comp_label] = {}
        emissivity = interpolator(y=model.emissivities[comp_label])(
            bandpass.frequencies.value
        )
        if model.albedos is not None:
            albedo = interpolator(y=model.albedos[comp_label])(
                bandpass.frequencies.value
            )
        else:
            albedo = 0

        if bandpass.frequencies.size > 1:
            emissivity = bandpass.integrate(emissivity)
            albedo = bandpass.integrate(albedo)

        source_parameters[comp_label]["emissivity"] = emissivity
        source_parameters[comp_label]["albedo"] = albedo

    if model.phase_coefficients is not None:
        phase_coefficients = interpolator(y=np.asarray(model.phase_coefficients))(
            bandpass.frequencies.value
        )
        phase_coefficients = interpolator(y=np.asarray(model.phase_coefficients))(
            bandpass.frequencies.value
        )
    else:
        phase_coefficients = np.repeat(
            np.zeros((3, 1)), repeats=bandpass.frequencies.size, axis=-1
        )

    if model.solar_irradiance is not None:
        solar_irradiance = interpolator(y=model.solar_irradiance)(
            bandpass.frequencies.value
        )
        solar_irradiance = u.Quantity(solar_irradiance, "MJy /sr").to_value(
            SPECIFIC_INTENSITY_UNITS, equivalencies=u.spectral()
        )
    else:
        solar_irradiance = 0

    if bandpass.frequencies.size > 1:
        phase_coefficients = bandpass.integrate(phase_coefficients)
        solar_irradiance = bandpass.integrate(solar_irradiance)
    source_parameters["common"] = {}
    source_parameters["common"]["phase_coefficients"] = tuple(phase_coefficients)
    source_parameters["common"]["solar_irradiance"] = solar_irradiance
    source_parameters["common"]["T_0"] = model.T_0
    source_parameters["common"]["delta"] = model.delta

    return source_parameters


def get_source_parameters_rmm(
    bandpass: Bandpass, model: RRM
) -> dict[ComponentLabel | str, dict[str, Any]]:
    if not bandpass.frequencies.unit.is_equivalent(model.spectrum.unit):
        bandpass.switch_convention()

    spectrum = (
        model.spectrum.to_value(u.Hz)
        if model.spectrum.unit.is_equivalent(u.Hz)
        else model.spectrum.to_value(u.micron)
    )

    source_parameters: dict[ComponentLabel | str, dict[str, Any]] = {}
    calibration = interp1d(x=spectrum, y=model.calibration, fill_value="extrapolate")(
        bandpass.frequencies.value
    )
    calibration = u.Quantity(calibration, u.MJy / u.AU).to_value(u.Jy / u.cm)

    if bandpass.frequencies.size > 1:
        calibration = bandpass.integrate(calibration)

    for comp_label in model.comps.keys():
        source_parameters[comp_label] = {}
        source_parameters[comp_label]["T_0"] = model.T_0[comp_label]
        source_parameters[comp_label]["delta"] = model.delta[comp_label]

    source_parameters["common"] = {"calibration": calibration}

    return source_parameters


SOURCE_PARAMS_MAPPING: dict[type[InterplanetaryDustModel], GetSourceParametersFn] = {
    Kelsall: get_source_parameters_kelsall_comp,
    RRM: get_source_parameters_rmm,
}
