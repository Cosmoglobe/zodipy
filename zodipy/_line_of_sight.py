from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zodipy._component_label import ComponentLabel

DISTANCE_TO_JUPITER = 5.2 # AU
EPS = np.finfo(float).eps

# Line of sight steps
line_of_sight_steps: dict[ComponentLabel, int] = {
    ComponentLabel.CLOUD: 200,
    ComponentLabel.BAND1: 200,
    ComponentLabel.BAND2: 250,
    ComponentLabel.BAND3: 200,
    ComponentLabel.RING: 125,
    ComponentLabel.FEATURE: 125,
}

# Line of sight steps
line_of_sight_cutoffs: dict[ComponentLabel, float] = {
    ComponentLabel.CLOUD: DISTANCE_TO_JUPITER,
    ComponentLabel.BAND1: DISTANCE_TO_JUPITER,
    ComponentLabel.BAND2: DISTANCE_TO_JUPITER,
    ComponentLabel.BAND3: DISTANCE_TO_JUPITER,
    ComponentLabel.RING: 3,
    ComponentLabel.FEATURE: 2,
}

def get_line_of_sight(
    component_label: ComponentLabel,
    observer_position: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
) -> tuple[float, NDArray[np.floating], int]:
    """
    Returns the start, stop and number of steps along the line of sights for an
    Interplanetary Dust component given the observer position and unit vectors
    corresponding to the pointing.
    """

    cutoff = line_of_sight_cutoffs[component_label]
    n_steps = line_of_sight_steps[component_label]
    stop = _get_line_of_sight_endpoints(cutoff, observer_position, unit_vectors)

    return EPS, stop, n_steps


def _get_line_of_sight_endpoints(
    r_cutoff: float,
    observer_position: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Computes the distance from the observer to the point along the line of
    sight which intersects the the sphere of length r_max around the Sun.
    """

    x, y, z = observer_position.flatten()

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
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
    c = r ** 2 - r_cutoff ** 2
    q = -0.5 * b * (1 + np.sqrt(b ** 2 - 4 * c) / np.abs(b))

    return np.maximum(q, c / q)
