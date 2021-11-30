import astropy.constants as const
import numpy as np

δ_K98 = 0.4668626
T_0 = 286

h = const.h.value
c = const.c.value
k_B = const.k_B.value


def blackbody_emission(T: np.ndarray, ν: float) -> np.ndarray:
    """Returns the blackbody emission for a temperature T and frequency ν.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    ν
        Frequency [Hz].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    term1 = (2 * h * ν ** 3) / c ** 2
    term2 = np.expm1((h * ν) / (k_B * T))

    return term1 / term2


def interplanetary_temperature(
    R: np.ndarray,
    T_0: float = T_0,
    δ: float = δ_K98,
) -> np.ndarray:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic coordinates.
    T_0
        Temperature of the solar system at R = 1 AU.
    δ
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature.
    """

    return T_0 * R ** -δ
