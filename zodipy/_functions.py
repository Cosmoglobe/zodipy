from typing import Union

import astropy.constants as const
import numpy as np


def blackbody_emission(
    T: Union[float, np.ndarray], freq: float
) -> Union[float, np.ndarray]:    
    """Returns the blackbody emission.
    
    Assumes the frequency to be in units of GHz.

    Parameters
    ----------
    freq : float
        Frequency [GHz].
    T : float:
        Temperature of the blackbody [K]. 
    Returns
    -------
    float
        Blackbody emission [W / m^2 Hz sr].
    """

    freq *= 1e9
    term1 = (2*const.h.value*freq**3) / const.c.value**2
    term2 = np.expm1((const.h.value*freq) / (const.k_B.value*T))

    return term1 / term2


def interplanetary_temperature(
    R: Union[float, np.ndarray], T_0: float = 286, delta: float = 0.4668626
) -> Union[float, np.ndarray]:
    """Returns the interplanetary temperature as a function of radius.
    
    Parameters
    ----------
    R : float
        Radial distance from the sun in ecliptic coordinates.
    T_0 : float, optional 
        Temperature of the solar system at R = 1 AU.
    delta : float, optional
        Powerlaw index.

    Returns
    -------
    float : 
        Interplanetary temperature.
    """

    return T_0 * R**-delta