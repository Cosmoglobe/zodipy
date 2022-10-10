from __future__ import annotations

import numpy as np
import numpy.typing as npt

EPS = float(np.finfo(float).eps)


def get_line_of_sight_endpoints(
    cutoff: float,
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]:
    """Returns the start and stop positions along the line of sights."""

    x, y, z = obs_pos.flatten()

    if cutoff < (r := np.sqrt(x**2 + y**2 + z**2)):
        raise ValueError(f"los_dist_cut is {cutoff} but observer_pos is {r}")

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lat = np.cos(lat)
    b = 2 * (x * cos_lat * np.cos(lon) + y * cos_lat * np.sin(lon))
    c = r**2 - cutoff**2

    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return EPS, np.maximum(q, c / q)
