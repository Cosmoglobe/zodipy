from __future__ import annotations
from functools import lru_cache
from typing import Callable
from scipy import interpolate


from numpy.typing import NDArray
import numpy as np

from zodipy._line_of_sight import EPS
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

    R_range = np.linspace(EPS, 15, 5000)
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
