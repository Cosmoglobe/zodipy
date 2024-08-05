from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from zodipy.component import ComponentLabel
from zodipy.component_params import RRM

if TYPE_CHECKING:
    import numpy.typing as npt


R_0 = np.finfo(np.float64).eps
R_KUIPER_BELT = 30
R_EARTH = 1
R_JUPITER = 5.2

DIRBE_CUTOFFS: dict[ComponentLabel, tuple[float | np.float64, float | np.float64]] = {
    ComponentLabel.CLOUD: (R_0, R_JUPITER),
    ComponentLabel.BAND1: (R_0, R_JUPITER),
    ComponentLabel.BAND2: (R_0, R_JUPITER),
    ComponentLabel.BAND3: (R_0, R_JUPITER),
    ComponentLabel.RING: (R_EARTH - 0.2, R_EARTH + 0.2),
    ComponentLabel.FEATURE: (R_EARTH - 0.3, R_EARTH + 0.3),
}

RRM_CUTOFFS: dict[ComponentLabel, tuple[float | np.float64, float | np.float64]] = {
    ComponentLabel.FAN: (R_0, RRM[ComponentLabel.FAN].R_outer),  # type: ignore
    ComponentLabel.INNER_NARROW_BAND: (
        RRM[ComponentLabel.INNER_NARROW_BAND].R_inner,  # type: ignore
        RRM[ComponentLabel.INNER_NARROW_BAND].R_outer,  # type: ignore
    ),
    ComponentLabel.OUTER_NARROW_BAND: (
        RRM[ComponentLabel.OUTER_NARROW_BAND].R_inner,  # type: ignore
        RRM[ComponentLabel.OUTER_NARROW_BAND].R_outer,  # type: ignore
    ),
    ComponentLabel.BROAD_BAND: (
        RRM[ComponentLabel.BROAD_BAND].R_inner,  # type: ignore
        RRM[ComponentLabel.BROAD_BAND].R_outer,  # type: ignore
    ),
    ComponentLabel.COMET: (
        RRM[ComponentLabel.COMET].R_inner,  # type: ignore
        RRM[ComponentLabel.COMET].R_outer,  # type: ignore
    ),
    ComponentLabel.INTERSTELLAR: (R_0, R_KUIPER_BELT),
    ComponentLabel.RING_RRM: DIRBE_CUTOFFS[ComponentLabel.RING],
    ComponentLabel.FEATURE_RRM: DIRBE_CUTOFFS[ComponentLabel.FEATURE],
}


COMPONENT_CUTOFFS = {**DIRBE_CUTOFFS, **RRM_CUTOFFS}


def integrate_leggauss(
    func: Callable[[float], npt.NDArray[np.float64]],
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Integrate a function using Gauss-Laguerre quadrature."""
    return np.squeeze(sum(func(x) * w for x, w in zip(points, weights)))


def get_sphere_intersection(
    obs_pos: npt.NDArray[np.float64],
    unit_vectors: npt.NDArray[np.float64],
    cutoff: float | np.float64,
) -> npt.NDArray[np.float64]:
    """Returns the distance from the observer to a heliocentric sphere with radius `cutoff`."""
    x_0, y_0, z_0 = obs_pos
    r_obs = np.sqrt(x_0**2 + y_0**2 + z_0**2)
    if (r_obs > cutoff).any():
        return np.full(obs_pos.shape[-1], np.finfo(float).eps)

    u_x, u_y, u_z = unit_vectors
    lon = np.arctan2(u_y, u_x)
    lat = np.arcsin(u_z)

    cos_lat = np.cos(lat)
    b = 2 * (x_0 * cos_lat * np.cos(lon) + y_0 * cos_lat * np.sin(lon))
    c = r_obs**2 - cutoff**2

    q = -0.5 * b * (1 + np.sqrt(b**2 - 4 * c) / np.abs(b))

    return np.maximum(q, c / q)


def get_line_of_sight_range(
    components: Iterable[ComponentLabel],
    unit_vectors: npt.NDArray[np.float64],
    obs_pos: npt.NDArray[np.float64],
) -> tuple[
    dict[ComponentLabel, npt.NDArray[np.float64]],
    dict[ComponentLabel, npt.NDArray[np.float64]],
]:
    """Return the component-wise start- and end-point of each line of sight."""
    start = {
        comp: get_sphere_intersection(obs_pos, unit_vectors, COMPONENT_CUTOFFS[comp][0])
        for comp in components
    }
    stop = {
        comp: get_sphere_intersection(obs_pos, unit_vectors, COMPONENT_CUTOFFS[comp][1])
        for comp in components
    }
    return start, stop
