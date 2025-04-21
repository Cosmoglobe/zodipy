from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, TypeVar, Union

import numpy as np
import numpy.typing as npt
from astropy import units
from scipy import integrate, interpolate

from zodipy.component import ComponentLabel
from zodipy.zodiacal_light_model import RRM, Kelsall, ZodiacalLightModel

CompParamDict = dict[ComponentLabel, dict[str, Any]]
CommonParamDict = dict[str, Any]
UnpackedModelDicts = tuple[CompParamDict, CommonParamDict]
T = TypeVar("T", bound=ZodiacalLightModel)
UnpackModelCallable = Callable[
    [units.Quantity, Union[units.Quantity, None], T, bool], UnpackedModelDicts
]


def interp_and_unpack_kelsall(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: Kelsall,
    bounds_error: bool,
) -> UnpackedModelDicts:
    """InterplantaryDustModelToDicts implementation for Kelsall model."""
    model_spectrum = deepcopy(model.spectrum)
    wavelengths = wavelengths.to(model.spectrum.unit, equivalencies=units.spectral())

    comp_params: dict[ComponentLabel, dict[str, Any]] = {}
    common_params: dict[str, Any] = {
        "T_0": model.T_0,
        "delta": model.delta,
    }

    for comp_label in model.comps:
        comp_params[comp_label] = {}
        comp_params[comp_label]["emissivity"] = interp_spectral_param(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.emissivities[comp_label],
            bounds_error=bounds_error,
        )
        if model.albedos is not None:
            comp_params[comp_label]["albedo"] = interp_spectral_param(
                wavelengths,
                weights,
                model_spectrum,
                spectral_parameter=model.albedos[comp_label],
                bounds_error=bounds_error,
            )
        else:
            comp_params[comp_label]["albedo"] = 0

    common_params["C1"] = (
        interp_spectral_param(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.C1,
            use_nearest=True,
            bounds_error=bounds_error,
        )
        if model.C1 is not None
        else 0
    )
    common_params["C2"] = (
        interp_spectral_param(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.C2,
            use_nearest=True,
            bounds_error=bounds_error,
        )
        if model.C2 is not None
        else 0
    )

    common_params["C3"] = (
        interp_spectral_param(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.C3,
            use_nearest=True,
            bounds_error=bounds_error,
        )
        if model.C3 is not None
        else 0
    )
    if model.solar_irradiance is None:
        common_params["solar_irradiance"] = 0
    else:
        common_params["solar_irradiance"] = interp_spectral_param(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.solar_irradiance,
            bounds_error=bounds_error,
        )

    return comp_params, common_params


def interp_and_unpack_rrm(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: RRM,
    bounds_error: bool,
) -> UnpackedModelDicts:
    """InterplantaryDustModelToDicts implementation for Kelsall model."""
    model_spectrum = deepcopy(model.spectrum)
    wavelengths = wavelengths.to(model.spectrum.unit, equivalencies=units.spectral())

    comp_params: dict[ComponentLabel, dict[str, Any]] = {}
    common_params: dict[str, Any] = {}

    for comp_label in model.comps:
        comp_params[comp_label] = {}
        comp_params[comp_label]["T_0"] = model.T_0[comp_label]
        comp_params[comp_label]["delta"] = model.delta[comp_label]

    calibration = interp_spectral_param(
        wavelengths,
        weights,
        model_spectrum,
        spectral_parameter=model.calibration,
        bounds_error=bounds_error,
    )
    calibration_quantity = units.Quantity(calibration, unit=units.MJy / units.AU)
    common_params["calibration"] = calibration_quantity.to_value(units.Jy / units.cm)
    return comp_params, common_params


def interp_spectral_param(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model_spectrum: units.Quantity,
    spectral_parameter: npt.ArrayLike,
    use_nearest: bool = False,
    bounds_error: bool = True,
) -> npt.NDArray:
    """Interpolate a spectral parameters."""
    parameter = np.asarray(spectral_parameter)

    if not np.array_equal(model_spectrum.value, np.sort(model_spectrum.value)):
        model_spectrum = np.flip(model_spectrum)
        parameter = np.flip(parameter)

    fill_value = np.nan if bounds_error else "extrapolate"

    if use_nearest:
        interp_func = interpolate.interp1d(
            model_spectrum.value,
            parameter,
            bounds_error=bounds_error,
            fill_value=fill_value,
            kind="nearest",
        )
    else:
        interp_func = interpolate.interp1d(
            model_spectrum.value, parameter, bounds_error=bounds_error, fill_value=fill_value
        )

    interped_param = interp_func(wavelengths.value)
    if weights is not None:
        return integrate.trapezoid(weights.value * interped_param, x=wavelengths.value)
    return interped_param


interp_and_unpack_func_mapping: dict[type[ZodiacalLightModel], UnpackModelCallable] = {
    Kelsall: interp_and_unpack_kelsall,
    RRM: interp_and_unpack_rrm,
}


def get_model_interp_func(
    model: ZodiacalLightModel,
) -> UnpackModelCallable:
    """Get the appropriate parameter unpacker for the model."""
    return interp_and_unpack_func_mapping[type(model)]
