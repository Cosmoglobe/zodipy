from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from ._constants import c, h, k_B

if TYPE_CHECKING:
    import numpy.typing as npt


def get_blackbody_emission(
    freq: float | npt.NDArray[np.float64], T: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the blackbody emission given a sequence of frequencies and temperatures.

    Args:
        freq: Frequency [Hz].
        T: Temperature of the blackbody [K].

    Returns:
        Blackbody emission [W / m^2 Hz sr].

    """
    term1 = (2 * h * freq**3) / c**2
    term2 = np.expm1((h * freq) / (k_B * T))
    return term1 / term2


def get_dust_grain_temperature(
    R: npt.NDArray[np.float64], T_0: float, delta: float
) -> npt.NDArray[np.float64]:
    """Return the dust grain temperature given a radial distance from the Sun.

    Args:
        R: Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
        T_0: Temperature of dust grains located 1 AU from the Sun [K].
        delta: Powerlaw index.

    Returns:
        Dust grain temperature [K].

    """
    return T_0 * R**-delta


def get_scattering_angle(
    R_los: float | npt.NDArray[np.float64],
    R_helio: npt.NDArray[np.float64],
    X_los: npt.NDArray[np.float64],
    X_helio: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the scattering angle between the Sun and a point along the line of sight.

    Args:
        R_los: Distance from observer to a point along the line of sight.
        R_helio: Distance from the Sun to a point along the line of sight.
        X_los: Vector pointing from the observer to a point along the line of sight.
        X_helio: Heliocentric position of point along the line of sight.

    Returns:
        Scattering angle.

    """
    cos_theta = (X_los * X_helio).sum(axis=0) / (R_los * R_helio)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(-cos_theta)


def get_phase_function(
    Theta: npt.NDArray[np.float64], C: tuple[float, ...]
) -> npt.NDArray[np.float64]:
    """Return the phase function.

    Args:
        Theta: Scattering angle.
        C: Phase function parameters.

    Returns:
        The Phase funciton.

    """
    phase_normalization = _get_phase_normalization(C)
    return phase_normalization * (C[0] + C[1] * Theta + np.exp(C[2] * Theta))


@lru_cache
def _get_phase_normalization(C: tuple[float, ...]) -> float:
    """Return the analyitcal integral for the phase normalization factor N."""
    int_term1 = 2 * np.pi
    int_term2 = 2 * C[0]
    int_term3 = np.pi * C[1]
    int_term4 = (np.exp(C[2] * np.pi) + 1) / (C[2] ** 2 + 1)
    return 1 / (int_term1 * (int_term2 + int_term3 + int_term4))
