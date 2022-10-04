from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._ipd_model import InterplanetaryDustModel
from .ipd_models import model_registry

DEFAULT_EARTH_POS = (1, 0, 0)


def tabulate_density(
    grid: NDArray[np.floating] | list[NDArray[np.floating]],
    model: str | InterplanetaryDustModel = "DIRBE",
    earth_position: tuple[float, float, float]
    | NDArray[np.floating] = DEFAULT_EARTH_POS,
) -> NDArray[np.floating]:
    """Returns the tabulated densities of the Interplanetary Dust components.

    Parameters
    ----------
    grid
        A cartesian mesh grid (x, y, z) created with `np.meshgrid` for which to
        tabulate the Interplanetary dust components.
    model
        The model who's Interplanetary Dust components to tabulate.
    earth_position
        The position of the Earth.

    Returns
    -------
    density_grid
        The tabulate densities of the Interplanetary Dust components.
    """

    if not isinstance(model, InterplanetaryDustModel):
        model = model_registry.get_model(model)

    if not isinstance(grid, np.ndarray):
        grid = np.asarray(grid)

    earth_position = np.reshape(earth_position, (3, 1, 1, 1))

    density_grid = np.zeros((model.n_comps, *grid.shape[1:]))
    for idx, comp in enumerate(model.comps.values()):
        comp.X_0 = np.reshape(comp.X_0, (3, 1, 1, 1))
        density_grid[idx] = comp.compute_density(
            X_helio=grid,
            X_earth=earth_position,
        )
        comp.X_0 = np.reshape(comp.X_0, (3, 1))

    return density_grid
