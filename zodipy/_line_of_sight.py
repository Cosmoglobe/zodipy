from __future__ import annotations
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from zodipy._component_label import ComponentLabel


R_JUPITER = 5.2  # AU
EPS = np.finfo(float).eps

# Line of sight steps
line_of_sight_steps: dict[ComponentLabel, int] = {
    ComponentLabel.CLOUD: 500,
    ComponentLabel.BAND1: 500,
    ComponentLabel.BAND2: 500,
    ComponentLabel.BAND3: 500,
    ComponentLabel.RING: 500,
    ComponentLabel.FEATURE: 500,
}

# Line of sight steps
line_of_sight_cutoffs: dict[ComponentLabel, float] = {
    ComponentLabel.CLOUD: R_JUPITER,
    ComponentLabel.BAND1: R_JUPITER,
    ComponentLabel.BAND2: R_JUPITER,
    ComponentLabel.BAND3: R_JUPITER,
    ComponentLabel.RING: 3,
    ComponentLabel.FEATURE: 2,
}


class LineOfSight(NamedTuple):
    """Line of sight info."""

    r_min: float
    r_max: NDArray[np.floating]
    n_steps: int

    @property
    def dr(self) -> NDArray[np.floating]:
        return (self.r_max - self.r_min) / self.n_steps

    @classmethod
    def from_comp_label(
        cls,
        comp_label: ComponentLabel,
        obs_pos: NDArray[np.floating],
        unit_vectors: NDArray[np.floating],
    ) -> LineOfSight:
        """Returns a LineOfSight object for a given component.
        
        Parameters
        ----------
        comp_label
            Label refering to an IPD component.
        obs_pos
            Position of the Solar System observer.
        unit_vectors
            Unit vectors associated with the observers pointing.
        """

        n_steps = line_of_sight_steps[comp_label]
        cut_off = line_of_sight_cutoffs[comp_label]

        r_max = get_line_of_sight_range(
            r_cutoff=cut_off,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
        )

        return cls(r_min=EPS, r_max=r_max, n_steps=n_steps)


def get_line_of_sight_range(
    r_cutoff: float,
    obs_pos: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Computes the distance from the observer to the point along the line of
    sight which intersects the the sphere of length r_max around the Sun.
    """

    x, y, z = obs_pos.flatten()

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
