import astropy.constants as const
import numpy as np

DELTA = 0.4668626
T_0 = 286


def blackbody_emission(T: np.ndarray, freq: float) -> np.ndarray:    
    """Returns the blackbody emission.
    
    Assumes the frequency to be in units of GHz.

    Parameters
    ----------
    freq
        Frequency in GHz.
    T
        Temperature of the blackbody in Kelvin. 

    Returns
    -------
        Blackbody emission in units of W / m^2 Hz sr.
    """

    freq *= 1e9
    term1 = (2*const.h.value*freq**3) / const.c.value**2
    term2 = np.expm1((const.h.value*freq) / (const.k_B.value*T))

    return term1 / term2


def interplanetary_temperature(
    R: np.ndarray, T_0: float = T_0, delta: float = DELTA
) -> np.ndarray:
    """Returns the interplanetary temperature as a function of radius.
    
    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic coordinates.
    T_0
        Temperature of the solar system at R = 1 AU.
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature.
    """

    return T_0 * R**-delta