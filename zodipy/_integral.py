from __future__ import annotations
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from zodipy._line_of_sight import LineOfSight


def trapezoidal(
    get_comp_step_emission: Callable[[float | NDArray[np.floating]], NDArray[np.floating]],
    line_of_sight: LineOfSight,
) -> NDArray[np.floating]:
    """Returns the integrated Zodiacal emission for an IPDcomponent

    This function implements the trapezoidal rule for a regular grid.

    Parameters
    ----------
    get_comp_step_emission
        Function that computes the Zodiacal emission at a step along the line
        of sight for an IPD component.
    line_of_sight
       Representation of discrete points along a line of sight.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated copmonent emission [W / Hz / m^2 / sr].
    """

    r_min, r_max, n_steps = line_of_sight
    dr = line_of_sight.dr

    integrated_emission = get_comp_step_emission(r_min)
    integrated_emission += get_comp_step_emission(r_max)
    integrated_emission += 2 * sum(
        get_comp_step_emission(dr * step) for step in range(1, n_steps)
    )

    return integrated_emission * (dr / 2)
