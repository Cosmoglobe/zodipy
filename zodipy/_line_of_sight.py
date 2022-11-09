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
)
from zodipy._ipd_comps import ComponentLabel

# Mapping of components to their inner and outer cutoff. None means that there is no
# inner cutoff and that the line of sight starts at the observer.
COMPONENT_CUTOFFS: dict[ComponentLabel, float] = {
    ComponentLabel.CLOUD: R_JUPITER,
    ComponentLabel.BAND1: R_JUPITER,
    ComponentLabel.BAND2: R_JUPITER,
    ComponentLabel.BAND3: R_JUPITER,
    ComponentLabel.RING: R_JUPITER,
    ComponentLabel.FEATURE: R_JUPITER,
    ComponentLabel.FAN: R_MARS,
    ComponentLabel.COMET: R_KUIPER_BELT,
    ComponentLabel.INTERSTELLAR: R_KUIPER_BELT,
    ComponentLabel.INNER_NARROW_BAND: R_THEMIS,
    ComponentLabel.OUTER_NARROW_BAND: R_EOS,
    ComponentLabel.BROAD_BAND: R_ASTEROID_BELT,
    ComponentLabel.RING_RRM: R_JUPITER,
    ComponentLabel.FEATURE_RRM: R_JUPITER,
}


def get_distance_from_obs_to_cutoff(
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
    cutoff: float,
) -> npt.NDArray[np.float64]:

    x, y, z = obs_pos.flatten()
    r = np.sqrt(x**2 + y**2 + z**2)

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lat = np.cos(lat)
    b = 2 * (x * cos_lat * np.cos(lon) + y * cos_lat * np.sin(lon))
    c = -np.abs(r**2 - cutoff**2)

    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return np.maximum(q, c / q)


def get_radial_line_of_sight_cutoff(
    components: Iterable[ComponentLabel],
    unit_vectors: npt.NDArray[np.float64],
    obs_pos: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    start = np.asarray([np.finfo(float).eps])
    cutoffs = [COMPONENT_CUTOFFS[component] for component in components]
    if len(set(cutoffs)) == 1:
        stop = get_distance_from_obs_to_cutoff(
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            cutoff=cutoffs[1],
        )
    else:
        stop = np.asarray(
            [
                get_distance_from_obs_to_cutoff(obs_pos, unit_vectors, cutoff)
                for cutoff in cutoffs
            ]
        )

    return start, stop
