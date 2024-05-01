from __future__ import annotations

from typing import Any, Callable, TypeVar, Union

import numpy as np
import numpy.typing as npt
from astropy import units
from scipy import integrate

from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipy.comps import ComponentLabel

CompParamDict = dict[ComponentLabel, dict[str, Any]]
CommonParamDict = dict[str, Any]


def kelsall_params_to_dicts(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: Kelsall,
) -> tuple[CompParamDict, CommonParamDict]:
    """InterplantaryDustModelToDicts implementation for Kelsall model."""
    model_spectrum = model.spectrum.to(wavelengths.unit, equivalencies=units.spectral())

    comp_params: dict[ComponentLabel, dict[str, Any]] = {}
    common_params: dict[str, Any] = {
        "T_0": model.T_0,
        "delta": model.delta,
    }

    for comp_label in model.comps:
        comp_params[comp_label] = {}
        comp_params[comp_label]["emissivity"] = interpolate_spectral_parameter(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.emissivities[comp_label],
        )
        if model.albedos is not None:
            comp_params[comp_label]["albedo"] = interpolate_spectral_parameter(
                wavelengths,
                weights,
                model_spectrum,
                spectral_parameter=model.albedos[comp_label],
            )
        else:
            comp_params[comp_label]["albedo"] = 0

    common_params["C1"] = (
        interpolate_spectral_parameter(
            wavelengths, weights, model_spectrum, spectral_parameter=model.C1
        )
        if model.C1 is not None
        else 0
    )
    common_params["C2"] = (
        interpolate_spectral_parameter(
            wavelengths, weights, model_spectrum, spectral_parameter=model.C2
        )
        if model.C2 is not None
        else 0
    )

    common_params["C3"] = (
        interpolate_spectral_parameter(
            wavelengths, weights, model_spectrum, spectral_parameter=model.C3
        )
        if model.C3 is not None
        else 0
    )

    if model.solar_irradiance is None:
        common_params["solar_irradiance"] = 0
    else:
        common_params["solar_irradiance"] = interpolate_spectral_parameter(
            wavelengths,
            weights,
            model_spectrum,
            spectral_parameter=model.solar_irradiance,
        )

    return comp_params, common_params


def rrm_params_to_dicts(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: RRM,
) -> tuple[CompParamDict, CommonParamDict]:
    """InterplantaryDustModelToDicts implementation for Kelsall model."""
    model_spectrum = model.spectrum.to(wavelengths.unit, equivalencies=units.spectral())

    comp_params: dict[ComponentLabel, dict[str, Any]] = {}
    common_params: dict[str, Any] = {}

    for comp_label in model.comps:
        comp_params[comp_label] = {}
        comp_params[comp_label]["T_0"] = model.T_0[comp_label]
        comp_params[comp_label]["delta"] = model.delta[comp_label]

    calibration = interpolate_spectral_parameter(
        wavelengths, weights, model_spectrum, spectral_parameter=model.calibration
    )
    calibration_quantity = units.Quantity(calibration, unit=units.MJy / units.AU)
    common_params["calibration"] = calibration_quantity.to_value(units.Jy / units.cm)
    return comp_params, common_params


def interpolate_spectral_parameter(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model_spectrum: units.Quantity,
    spectral_parameter: npt.ArrayLike,
) -> npt.NDArray:
    """Interpolate a spectral parameters."""
    paramameter = np.asarray(spectral_parameter)
    interpolated_parameter = np.interp(wavelengths.value, model_spectrum.value, paramameter)

    if weights is not None:
        return integrate.trapezoid(weights.value * interpolated_parameter, x=wavelengths.value)
    return interpolated_parameter


T = TypeVar("T", contravariant=True, bound=InterplanetaryDustModel)
CallableModelToDicts = Callable[
    [units.Quantity, Union[units.Quantity, None], T], tuple[CompParamDict, CommonParamDict]
]


MODEL_INTERPOLATION_MAPPING: dict[type[InterplanetaryDustModel], CallableModelToDicts] = {
    Kelsall: kelsall_params_to_dicts,
    RRM: rrm_params_to_dicts,
}


def get_model_to_dicts_callable(
    model: InterplanetaryDustModel,
) -> CallableModelToDicts:
    """Get the appropriate parameter unpacker for the model."""
    return MODEL_INTERPOLATION_MAPPING[type(model)]
