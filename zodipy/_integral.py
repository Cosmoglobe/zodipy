from typing import Callable

import numpy as np
from numpy.typing import NDArray


def trapezoidal(
    step_emission_func: Callable[[float], NDArray[np.floating]],
    line_of_sight: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method.

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

    emission_previous = step_emission_func(line_of_sight[0])
    integrated_emission = np.zeros_like(emission_previous)

    for r, dr in zip(line_of_sight[1:], np.diff(line_of_sight)):
        emission_current = step_emission_func(r)

        integrated_emission += (emission_previous + emission_current) * (dr / 2)

        emission_previous = emission_current

    return integrated_emission