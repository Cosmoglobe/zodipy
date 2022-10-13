from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt

from ._ipd_dens_funcs import construct_density_partials
from ._ipd_model import InterplanetaryDustModel
from .ipd_models import model_registry

DEFAULT_EARTH_POS = (1, 0, 0)


def tabulate_density(
    grid: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
    model: str | InterplanetaryDustModel = "DIRBE",
    earth_position: tuple[float, float, float]
    | npt.NDArray[np.float64] = DEFAULT_EARTH_POS,
) -> npt.NDArray[np.float64]:
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

    # Prepare attributes and variables for broadcasting with the grid
    earth_position = np.reshape(earth_position, (3, 1, 1, 1))
    for comp in model.comps.values():
        comp.X_0 = np.reshape(comp.X_0, (3, 1, 1, 1))

    partials = construct_density_partials(
        list(model.comps.values()), {"X_earth": earth_position}
    )

    density_grid = np.zeros((model.n_comps, *grid.shape[1:]))
    for idx, partial in enumerate(partials):
        density_grid[idx] = partial(grid)

    # Revert broadcasting reshapes
    for comp in model.comps.values():
        comp.X_0 = np.reshape(comp.X_0, (3, 1))

    return density_grid
