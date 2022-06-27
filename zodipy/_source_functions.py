from __future__ import annotations

from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

h = const.h.value
c = const.c.value
k_B = const.k_B.value
T_sun = 5778  # K

SPECIFIC_INTENSITY_UNITS = u.W / u.Hz / u.m**2 / u.sr


def get_blackbody_emission(
    freq: float | NDArray[np.floating], T: float | NDArray[np.floating]
) -> float | NDArray[np.floating]:
    """Returns the blackbody emission given a frequency.

    Parameters
    ----------
    freq
        Frequency [GHz].
    T
        Temperature of the blackbody [K].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    freq *= 1e9
    term1 = (2 * h * freq**3) / c**2
    term2 = np.expm1(((h * freq) / (k_B * T)), dtype=np.float128)

    return term1 / term2


def get_dust_grain_temperature(
    R: float | NDArray[np.floating], T_0: float, delta: float
) -> float | NDArray[np.floating]:
    """Returns the dust grain temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
    T_0
        Temperature of dust grains located 1 AU from the Sun [K].
    delta
        Powerlaw index.

    Returns
    -------
        Dust grain temperature [K].
    """

    return T_0 * R**-delta


def get_scattering_angle(
    R_los: float | NDArray[np.floating],
    R_helio: NDArray[np.floating],
    X_los: NDArray[np.floating],
    X_helio: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Returns the scattering angle between the Sun and a point along the
    line of sight.

    Parameters
    ----------
    R_los
        Distance from observer to a point along the line of sight.
    R_helio
        Distance from the Sun to a point along the line of sight.
    X_los
        vector pointing from the observer to a point along the line of sight.
    X_helio
        Heliocentric position of point along the line of sight.

    Returns
    -------
        Scattering angle.
    """
    cos_theta = (X_los * X_helio).sum(axis=0) / (R_los * R_helio)
    cos_theta = np.clip(cos_theta, -1, 1)

    return np.arccos(-cos_theta)


def get_phase_function(
    Theta: NDArray[np.floating], C: tuple[float, float, float]
) -> NDArray[np.floating]:
    """Returns the phase function.

    Parameters
    ----------
    Theta
        Scattering angle.
    coeffs
        Phase function parameters.

    Returns
    -------
        The Phase funciton.
    """

    phase_normalization = _get_phase_normalization(C)

    return phase_normalization * (C[0] + C[1] * Theta + np.exp(C[2] * Theta))


@lru_cache
def _get_phase_normalization(C: tuple[float, float, float]) -> float:
    """Returns the analyitcal integral for the phase normalization factor N."""

    int_term1 = 2 * np.pi
    int_term2 = 2 * C[0]
    int_term3 = np.pi * C[1]
    int_term4 = (np.exp(C[2] * np.pi) + 1) / (C[2] ** 2 + 1)

    return 1 / (int_term1 * (int_term2 + int_term3 + int_term4))
