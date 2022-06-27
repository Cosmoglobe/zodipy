from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = float(np.finfo(float).eps)


def get_line_of_sight_endpoints(
    cutoff: float, obs_pos: NDArray[np.floating], unit_vectors: NDArray[np.floating]
) -> tuple[float, NDArray[np.floating]]:
    """Returns the start and stop positions along the line of sights."""

    x, y, z = obs_pos.flatten()

    r = np.sqrt(x**2 + y**2 + z**2)
    if cutoff < r:
        raise ValueError(f"los_dist_cut is {cutoff} but observer_pos is {r}")

    theta = np.arctan2(y, x)

    x_0 = r * np.cos(theta)
    y_0 = r * np.sin(theta)

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lat = np.cos(lat)
    b = 2 * (x_0 * cos_lat * np.cos(lon) + y_0 * cos_lat * np.sin(lon))
    c = r**2 - cutoff**2

    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return EPS, np.maximum(q, c / q)
