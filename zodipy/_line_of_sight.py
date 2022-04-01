from __future__ import annotations
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from zodipy._labels import CompLabel


R_JUPITER = 5.2  # AU


# Line of sight steps
line_of_sight_steps: dict[CompLabel, int] = {
    CompLabel.CLOUD: 500,
    CompLabel.BAND1: 500,
    CompLabel.BAND2: 500,
    CompLabel.BAND3: 500,
    CompLabel.RING: 500,
    CompLabel.FEATURE: 500,
}

# Line of sight steps
line_of_sight_cutoffs: dict[CompLabel, float] = {
    CompLabel.CLOUD: R_JUPITER,
    CompLabel.BAND1: R_JUPITER,
    CompLabel.BAND2: R_JUPITER,
    CompLabel.BAND3: R_JUPITER,
    CompLabel.RING: 3,
    CompLabel.FEATURE: 2,
}


class LineOfSight(NamedTuple):
    """Line of sight info."""

    n_steps: int
    r_max: NDArray[np.floating]
    r_min: float = np.finfo(float).eps

    @property
    def dr(self) -> NDArray[np.floating]:
        return (self.r_max - self.r_min) / self.n_steps


    @classmethod
    def from_comp_label(
        cls,
        comp_label: CompLabel,
        obs_pos: tuple[float, float, float],
        unit_vectors: NDArray[np.floating],
    ) -> LineOfSight:

        n_steps = line_of_sight_steps[comp_label]
        cut_off = line_of_sight_cutoffs[comp_label]

        r_max = get_line_of_sight_range(
            r_cutoff=cut_off,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
        )

        return LineOfSight(n_steps, r_max)


def get_line_of_sight_range(
    r_cutoff: float,
    obs_pos: tuple[float, float, float],
    unit_vectors: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Computes the distance from the observer to the point along the line of
    sight which intersects the the sphere of length r_max around the Sun.
    """

    x, y, z = obs_pos
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

    r_los = np.maximum(q, c / q)

    return r_los
