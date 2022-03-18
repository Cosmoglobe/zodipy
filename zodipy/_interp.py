from __future__ import annotations
from functools import lru_cache
from typing import Callable, Optional, Sequence
from scipy import interpolate

import astropy.units as u
from astropy.units import Quantity
from numpy.typing import NDArray
import numpy as np

from zodipy._labels import CompLabel
from zodipy._los_config import EPS
from zodipy._source_funcs import (
    blackbody_emission_nu,
    interplanetary_temperature,
    R_sun,
    T_sun,
)


@lru_cache
def tabulated_blackbody_emission_nu(
    freq: float,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Returns tabulated, cached array of blackbody emission."""

    T_range = np.linspace(50, 10000, 5000)
    tabulated_bnu = blackbody_emission_nu(T=T_range, freq=freq)

    return interpolate.interp1d(T_range, tabulated_bnu)


def interp_blackbody_emission_nu(
    freq: float, T: float | NDArray[np.floating]
) -> NDArray[np.floating]:
    """Returns the interpolated black body emission for a temperature."""

    f = tabulated_blackbody_emission_nu(freq=freq)
    try:
        return f(T)
    except ValueError:
        return blackbody_emission_nu(T=T, freq=freq)


@lru_cache
def tabulated_interplanetary_temperature(
    T_0: float, delta: float
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Returns tabulated, cached array of interplanetary temperatures."""

    R_range = np.linspace(EPS.value, 15, 5000)
    tabulated_T = interplanetary_temperature(R_range, T_0, delta)

    return interpolate.interp1d(R_range, tabulated_T)


def interp_interplanetary_temperature(
    R: NDArray[np.floating], T_0: float, delta: float
) -> NDArray[np.floating]:
    """Returns the intepolated interplanetary temperature."""

    f = tabulated_interplanetary_temperature(T_0=T_0, delta=delta)

    return f(R)


def interp_solar_flux(
    R: NDArray[np.floating], freq: float, T: float = T_sun
) -> NDArray[np.floating]:
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
    freq: Quantity[u.GHz],
    emissivities: dict[CompLabel, Sequence[float]],
    emissivity_spectrum: Quantity[u.Hz] | Quantity[u.m],
    albedos: Optional[dict[CompLabel, Sequence[float]]],
    albedo_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]],
) -> dict[CompLabel, dict[str, float]]:

    interp_params: dict[CompLabel, dict[str, float]] = {
        comp_label: {} for comp_label in emissivities
    }

    for comp, emiss_Sequence in emissivities.items():
        interp_params[comp]["emissivity"] = np.interp(
            freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
            emissivity_spectrum,
            emiss_Sequence,
        )

    if albedos is not None and albedo_spectrum is not None:
        for comp, albedo_Sequence in albedos.items():
            interp_params[comp]["albedo"] = np.interp(
                freq.to(albedo_spectrum.unit, equivalencies=u.spectral()),
                albedo_spectrum,
                albedo_Sequence,
            )
    else:
        for comp in emissivities:
            interp_params[comp]["albedo"] = 0.0

    return interp_params


def interp_phase_coeffs(
    freq: Quantity[u.GHz],
    phase_coeffs: Optional[dict[str, Quantity]] = None,
    phase_coeffs_spectrum: Optional[Quantity[u.Hz] | Quantity[u.m]] = None,
) -> Sequence[float]:

    if phase_coeffs is not None and phase_coeffs_spectrum is not None:
        interp_phase_coeffs = [
            np.interp(
                freq.to(phase_coeffs_spectrum.unit, equivalencies=u.spectral()),
                phase_coeffs_spectrum,
                coeff,
            ).value
            for coeff in phase_coeffs.values()
        ]
    else:
        interp_phase_coeffs = [0.0 for _ in range(3)]

    return interp_phase_coeffs
