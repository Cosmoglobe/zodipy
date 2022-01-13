from typing import Sequence, Tuple, Union
import astropy.constants as const
import numpy as np


h = const.h.value
c = const.c.value
k_B = const.k_B.value


def blackbody_emission(T: Union[float, np.ndarray], freq: float) -> np.ndarray:
    """Returns the blackbody emission for a temperature T and frequency freq.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    freq
        Frequency [GHz].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    freq *= 1e9
    term1 = (2 * h * freq ** 3) / c ** 2
    term2 = np.expm1((h * freq) / (k_B * T))

    return term1 / term2


def interplanetary_temperature(R: np.ndarray, T_0: float, delta: float) -> np.ndarray:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates.
    T_0
        Temperature of the solar system at R = 1 AU.
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature.
    """

    return T_0 * R ** -delta


def phase_function(Theta: np.ndarray, C: Sequence[float]) -> np.ndarray:
    """Returns the phase function.

    Parameters
    ----------
    Theta
        Scattering angle [rad].
    C
        Phase function parameters.

    Returns
    -------
        The Phase funciton.
    """

    phase = C[0] + C[1] * Theta + np.exp(C[2] * Theta)
    N = 1 / (phase.sum())

    return N * phase
