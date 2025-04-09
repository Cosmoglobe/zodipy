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
UnpackModelCallable = Callable[[units.Quantity, Union[units.Quantity, None], T], UnpackedModelDicts]


def interp_and_unpack_kelsall(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: Kelsall,
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
        )
        if model.albedos is not None:
            comp_params[comp_label]["albedo"] = interp_spectral_param(
                wavelengths,
                weights,
                model_spectrum,
                spectral_parameter=model.albedos[comp_label],
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
        )

    return comp_params, common_params


def interp_and_unpack_rrm(
    wavelengths: units.Quantity,
    weights: units.Quantity | None,
    model: RRM,
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
        wavelengths, weights, model_spectrum, spectral_parameter=model.calibration
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
) -> npt.NDArray:
    """Interpolate a spectral parameters."""
    paramameter = np.asarray(spectral_parameter)

    if not np.array_equal(model_spectrum.value, np.sort(model_spectrum.value)):
        model_spectrum = np.flip(model_spectrum)
        paramameter = np.flip(paramameter)

    if use_nearest:
        interped_param = interpolate.interp1d(model_spectrum.value, paramameter, kind="nearest")(
            wavelengths.value
        )
    else:
        interped_param = np.interp(wavelengths.value, model_spectrum.value, paramameter)

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
