from __future__ import annotations

from typing import Callable, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall

TModel = TypeVar("TModel", bound=InterplanetaryDustModel)
GetCutoffFn = Callable[
    [TModel, npt.NDArray, npt.NDArray], Tuple[npt.NDArray, npt.NDArray]
]


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


def get_single_outer_cutoff(
    model: Kelsall,
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    start = np.asarray([np.finfo(float).eps])
    stop = get_distance_from_obs_to_cutoff(
        obs_pos=obs_pos,
        unit_vectors=unit_vectors,
        cutoff=model.outer_cutoff,
    )

    return start, stop


def get_multiple_inner_and_outer_cutoff(
    model: RRM,
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    start_list: list[npt.NDArray[np.float64]] = []
    for inner_cutoff in model.inner_cutoff.values():
        if inner_cutoff is None:
            start_list.append(np.full(np.shape(unit_vectors)[-1], np.finfo(float).eps))
        else:
            start_list.append(
                get_distance_from_obs_to_cutoff(
                    obs_pos=obs_pos,
                    unit_vectors=unit_vectors,
                    cutoff=inner_cutoff,
                )
            )

    stop_list = [
        get_distance_from_obs_to_cutoff(obs_pos, unit_vectors, outer_cutoff)
        for outer_cutoff in model.outer_cutoff.values()
    ]

    return np.asarray(start_list), np.asarray(stop_list)


LOS_MAPPING: dict[type[InterplanetaryDustModel], GetCutoffFn] = {
    RRM: get_multiple_inner_and_outer_cutoff,
    Kelsall: get_single_outer_cutoff,
}
