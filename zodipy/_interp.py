from functools import lru_cache
from typing import Any, Callable, Dict, Union
from scipy import interpolate

import astropy.units as u
from astropy.units import Quantity
from numpy.typing import NDArray
import numpy as np

from zodipy._labels import Label
from zodipy._integration_config import EPS
from zodipy._source_functions import (
    blackbody_emission_nu,
    interplanetary_temperature,
    R_sun,
    T_sun,
)


@lru_cache
def tabulated_blackbody_emission_nu(
    freq: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Returns tabulated, cached array of blackbody emission."""

    T_range = np.linspace(50, 10000, 5000)
    tabulated_bnu = blackbody_emission_nu(T=T_range, freq=freq)

    return interpolate.interp1d(T_range, tabulated_bnu)


def interp_blackbody_emission_nu(
    freq: float, T: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Returns the interpolated black body emission for a temperature."""

    f = tabulated_blackbody_emission_nu(freq=freq)
    try:
        return f(T)
    except ValueError:
        return blackbody_emission_nu(T=T, freq=freq)


@lru_cache
def tabulated_interplanetary_temperature(
    T_0: float, delta: float
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Returns tabulated, cached array of interplanetary temperatures."""

    R_range = np.linspace(EPS.value, 15, 5000)
    tabulated_T = interplanetary_temperature(R_range, T_0, delta)

    return interpolate.interp1d(R_range, tabulated_T)


def interp_interplanetary_temperature(
    R: NDArray[np.float64], T_0: float, delta: float
) -> NDArray[np.float64]:
    """Returns the intepolated interplanetary temperature."""

    f = tabulated_interplanetary_temperature(T_0=T_0, delta=delta)

    return f(R)


def interp_solar_flux(
    R: NDArray[np.float64], freq: float, T: float = T_sun
) -> NDArray[np.float64]:
    """Returns the interpolated solar flux.

    Parameteers
    -----------
    freq
        Frequency [GHz].
    T
        Temperature [K].

    Returns
    -------
        Solar flux at some distance R from the Sun in AU.
    """

    return np.pi * interp_blackbody_emission_nu(T=T, freq=freq) * (R_sun / R) ** 2


def interp_comp_spectral_params(
    component: Label,
    freq: Quantity[u.GHz],
    spectral_params: Dict[Any, Any],
) -> Dict[Any, Any]:
    """Returns interpolated source parameters given a frequency and a component."""

    params: Dict[Any, Any] = {}
    emissivities = spectral_params.get("emissivities")
    if emissivities is not None:
        emissivity_spectrum = emissivities["spectrum"]
        params["emissivity"] = np.interp(
            freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
            emissivity_spectrum,
            emissivities[component],
        )
    else:
        params["emissivity"] = 1.0

    albedos = spectral_params.get("albedos")
    if albedos is not None:
        albedo_spectrum = albedos["spectrum"]
        params["albedo"] = np.interp(
            freq.to(albedo_spectrum.unit, equivalencies=u.spectral()),
            albedo_spectrum,
            albedos[component],
        )
    else:
        params["albedo"] = 0.0

    phases = spectral_params.get("phase")
    if phases is not None:
        phase_spectrum = phases["spectrum"]
        params["phase_coeffs"] = {
            coeff: np.interp(
                freq.to(phase_spectrum.unit, equivalencies=u.spectral()),
                phase_spectrum,
                phases[coeff],
            ).value
            for coeff in ["C0", "C1", "C2"]
        }
    else:
        params["phase_coeffs"] = {"C0": 0.0, "C1": 0.0, "C2": 0.0}

    return params
