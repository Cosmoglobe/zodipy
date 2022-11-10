from __future__ import annotations

from typing import Iterable

import numpy as np
import numpy.typing as npt

from zodipy._constants import (
    R_ASTEROID_BELT,
    R_EARTH,
    R_JUPITER,
    R_KUIPER_BELT,
    R_MARS,
    R_EOS,
    R_THEMIS,
    R_0,
)
from zodipy._ipd_comps import ComponentLabel

DIRBE_CUTOFFS: dict[ComponentLabel, tuple[float | np.float64, float]] = {
    ComponentLabel.CLOUD: (R_0, R_JUPITER),
    ComponentLabel.BAND1: (R_0, R_JUPITER),
    ComponentLabel.BAND2: (R_0, R_JUPITER),
    ComponentLabel.BAND3: (R_0, R_JUPITER),
    ComponentLabel.RING: (R_EARTH - 0.2, R_EARTH + 0.2),
    ComponentLabel.FEATURE: (R_EARTH - 0.2, R_EARTH + 0.2),
}

RRM_CUTOFFS: dict[ComponentLabel, tuple[float | np.float64, float]] = {
    ComponentLabel.FAN: (R_0, R_MARS),
    ComponentLabel.COMET: (R_MARS, R_KUIPER_BELT),
    ComponentLabel.INTERSTELLAR: (R_0, R_KUIPER_BELT),
    ComponentLabel.INNER_NARROW_BAND: (R_MARS, R_THEMIS),
    ComponentLabel.OUTER_NARROW_BAND: (R_MARS, R_EOS),
    ComponentLabel.BROAD_BAND: (R_MARS, R_ASTEROID_BELT),
    ComponentLabel.RING_RRM: DIRBE_CUTOFFS[ComponentLabel.RING],
    ComponentLabel.FEATURE_RRM: DIRBE_CUTOFFS[ComponentLabel.FEATURE],
}


COMPONENT_CUTOFFS = {**DIRBE_CUTOFFS, **RRM_CUTOFFS}


def get_sphere_intersection(
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
    cutoff: float | np.float64,
) -> npt.NDArray[np.float64]:
    """
    Given the observer position, return distance from the observer to the
    intersection between the line of sights and a heliocentric sphere with radius cutoff.
    """

    x, y, z = obs_pos.flatten()
    r_obs = np.sqrt(x**2 + y**2 + z**2)

    if r_obs > cutoff:
        return np.asarray([np.finfo(float).eps])

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lat = np.cos(lat)
    b = 2 * (x * cos_lat * np.cos(lon) + y * cos_lat * np.sin(lon))
    c = r_obs**2 - cutoff**2

    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return np.maximum(q, c / q)


def get_line_of_sight_start_and_stop_distances(
    components: Iterable[ComponentLabel],
    unit_vectors: npt.NDArray[np.float64],
    obs_pos: npt.NDArray[np.float64],
) -> tuple[
    dict[ComponentLabel, npt.NDArray[np.float64]],
    dict[ComponentLabel, npt.NDArray[np.float64]],
]:

    start = {
        comp: get_sphere_intersection(obs_pos, unit_vectors, COMPONENT_CUTOFFS[comp][0])
        for comp in components
    }
    stop = {
        comp: get_sphere_intersection(obs_pos, unit_vectors, COMPONENT_CUTOFFS[comp][1])
        for comp in components
    }
    return start, stop
