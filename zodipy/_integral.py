from __future__ import annotations
from typing import Callable

import numpy as np
from numpy.typing import NDArray


def trapezoidal_regular_grid(
    get_emission_step: Callable[[float | NDArray[np.floating]], NDArray[np.floating]],
    start: float,
    stop: float | NDArray[np.floating],
    n_steps: int,
) -> NDArray[np.floating]:
    """
    Integrates and returns the Zodiacal Emission of a Interplanetary Dust component
    over a regular grid.

    Parameters
    ----------
    get_emission_step
        Function that computes the Zodiacal emission at a step along the line
        of sight for an Interplanetary Dust component.
    start
        Lower integration limit (At the face of the observer).
    stop
        Upper integration limit (At a distance along the line of sight which
        corresponds to a heliocentric distance of 5.2 AU)
    step
        Number of steps along the line of sight to integrate.

    Returns
    -------
    integrated_emission
        Integrated Zodiacal emission for an Interplanetary Dust component over
        line of sights in units of W / Hz / m^2 / sr.
    """
    
    ds = (stop - start) / n_steps

    integrated_emission = get_emission_step(start) + get_emission_step(stop)
    integrated_emission += 2 * sum(
        get_emission_step(start + ds * step) for step in range(1, n_steps)
    )

    return integrated_emission * (ds / 2)
