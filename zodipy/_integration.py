from typing import Callable

import numpy as np

EmissionCallable = Callable[
    [float, np.ndarray, np.ndarray, np.ndarray, float], np.ndarray
]


def trapezoidal(
    emission_func: EmissionCallable,
    freq: float,
    x_obs: np.ndarray,
    x_earth: np.ndarray,
    x_unit: np.ndarray,
    R: np.ndarray,
    npix: int,
    pixels: np.ndarray,
) -> np.ndarray:
    """Integrates the emission for a component using the trapezoidal method."""

    comp_emission = np.zeros(npix)[pixels]
    emission_prev = emission_func(freq, x_obs, x_earth, x_unit, R[0])
    for i in range(1, len(R)):
        dR = R[i] - R[i - 1]
        emission_cur = emission_func(freq, x_obs, x_earth, x_unit, R[i])
        comp_emission += (emission_prev + emission_cur) * dR / 2
        emission_prev = emission_cur

    return comp_emission
