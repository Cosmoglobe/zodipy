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


def get_blackbody_emission_nu(
    freq: float | NDArray[np.floating],
    T: float | NDArray[np.floating],
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


def get_solar_flux(
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

    return np.pi * get_blackbody_emission_nu(T=T, freq=freq) * (R_sun / R) ** 2


def get_interplanetary_temperature(
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


def get_phase_function(
    Theta: NDArray[np.floating], phase_coefficients: tuple[float, float, float]
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

    c_0, c_1, c_2 = phase_coefficients
    phase_normalization = _get_phase_normalization(c_0, c_1, c_2)

    return phase_normalization * (c_0 + c_1 * Theta + np.exp(c_2 * Theta))


@lru_cache
def _get_phase_normalization(c_0: float, c_1: float, c_2: float) -> float:
    """Returns the analyitcal integral for the phase normalization factor N."""

    pi = np.pi
    int_term1 = 2 * pi
    int_term2 = 2 * c_0
    int_term3 = pi * c_1
    int_term4 = (np.exp(c_2 * pi) + 1) / (c_2 ** 2 + 1)

    return 1 / (int_term1 * (int_term2 + int_term3 + int_term4))
