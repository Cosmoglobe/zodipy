from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DISTANCE_TO_JUPITER = 5.2  # AU
EPS = float(np.finfo(float).eps)


def get_line_of_sight_start_stop(
    cutoff: float, obs_pos: NDArray[np.floating], unit_vectors: NDArray[np.floating]
) -> tuple[float, NDArray[np.floating]]:
    """
    Returns the start and stop positions along the line of sights for an
    Interplanetary Dust component given the observer position and unit vectors
    corresponding to the pointing.
    """

    stop = _get_line_of_sight_endpoints(cutoff, obs_pos, unit_vectors)

    return EPS, stop


def _get_line_of_sight_endpoints(
    cutoff: float, obs_pos: NDArray[np.floating], unit_vectors: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Computes the distance from the observer to the point along the line of
    sight which intersects the the sphere of length r_max around the Sun.
    """

    x, y, z = obs_pos.flatten()

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    x_0 = r * cos_theta
    y_0 = r * sin_theta

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    cos_lat = np.cos(lat)

    b = 2 * (x_0 * cos_lat * cos_lon + y_0 * cos_lat * sin_lon)
    c = r**2 - cutoff**2
    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return np.maximum(q, c / q)
