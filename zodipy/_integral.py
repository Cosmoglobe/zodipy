from __future__ import annotations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from zodipy._line_of_sight import LineOfSight


def trapezoidal(
    step_emission_func: Callable[[float | NDArray[np.floating]], NDArray[np.floating]],
    line_of_sight: LineOfSight,
) -> NDArray[np.floating]:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method for a regular line of sight grid.

    Parameters
    ----------
    step_emission_func
        Function that computes the Zodiacal emission at a step along the line
        of sight.
    line_of_sight
       Representation of discrete points along a line of sight.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated copmonent emission [W / Hz / m^2 / sr].
    """

    integrated_emission = step_emission_func(line_of_sight.r_min)
    integrated_emission += step_emission_func(line_of_sight.r_max)
    integrated_emission += 2 * sum(
        step_emission_func(line_of_sight.dr * step)
        for step in range(1, line_of_sight.n_steps)
    )

    return integrated_emission * (line_of_sight.dr / 2)
