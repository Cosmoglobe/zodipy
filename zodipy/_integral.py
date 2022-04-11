from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def trapezoidal_regular_grid(
    emission_step_function: Callable[[float | NDArray[np.floating]], NDArray[np.floating]],
    start: float,
    stop: float | NDArray[np.floating],
    n_steps: int,
) -> NDArray[np.floating]:
    """
    Integrates and returns the Zodiacal Emission of a Interplanetary Dust component
    using the trapezoidal method over a regular grid.

    Parameters
    ----------
    emission_step_function
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

    integrated_emission = emission_step_function(start) + emission_step_function(stop)
    integrated_emission += 2 * sum(
        emission_step_function(start + ds * step) for step in range(1, n_steps)
    )

    return integrated_emission * (ds / 2)
