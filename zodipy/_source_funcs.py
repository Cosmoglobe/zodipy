from __future__ import annotations
from functools import lru_cache

import astropy.constants as const
import astropy.units as u
import numpy as np
from numpy.typing import NDArray


h = const.h.value
c = const.c.value
k_B = const.k_B.value
R_sun = const.R_sun.to(u.AU).value
T_sun = 5778
π = np.pi


def blackbody_emission_nu(
    T: float | NDArray[np.floating], freq: float | NDArray[np.floating]
) -> NDArray[np.floating]:
    """Returns the blackbody emission.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    freq
        Frequency [GHz] .

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """
    freq *= 1e9
    term1 = (2 * h * freq ** 3) / c ** 2
    term2 = np.expm1(((h * freq) / (k_B * T)))

    return term1 / term2


def blackbody_emission_lambda(
    T: float | NDArray[np.floating],
    freq: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Returns the blackbody emission.

    Parameters
    ----------
    T
        Temperature of the blackbody.
    freq
        Frequency.

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """
    freq *= 1e-6
    term1 = (2 * h * c ** 2) / freq ** 5
    term2 = np.expm1(((h * c) / (freq * k_B * T)))

    return term1 / term2


def solar_flux(
    R: NDArray[np.floating], freq: float, T: float = T_sun
) -> NDArray[np.floating]:
    """Returns the solar flux observed at some distance R from the Sun in AU.

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

    return np.pi * blackbody_emission_nu(T=T, freq=freq) * (R_sun / R) ** 2


def interplanetary_temperature(
    R: NDArray[np.floating], T_0: float, delta: float
) -> NDArray[np.floating]:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
    T_0
        Temperature of the solar system at 1 AU [K].
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature [K].
    """

    return T_0 * R ** -delta


@lru_cache
def phase_normalization(C0: float, C1: float, C2: float) -> float:
    """Returns the analyitcal integral for the phase normalization factor N."""

    int_term1 = 2 * π
    int_term2 = 2 * C0
    int_term3 = π * C1
    int_term4 = (np.exp(C2 * π) + 1) / (C2 ** 2 + 1)

    return 1 / (int_term1 * (int_term2 + int_term3 + int_term4))


def phase_function(
    Theta: NDArray[np.floating], C0: float, C1: float, C2: float
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

    N = phase_normalization(C0=C0, C1=C1, C2=C2)

    return N * (C0 + C1 * Theta + np.exp(C2 * Theta)) 
