from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zodipy.models import model_registry
from zodipy._model import InterplanetaryDustModel
from zodipy._labels import CompLabel

DEFAULT_EARTH_POS = np.asarray([1.0, 0.0, 0.0])


def tabulate_density(
    grid: NDArray[np.float_] | list[NDArray[np.float_]],
    model: str | InterplanetaryDustModel = "DIRBE",
    earth_coords: NDArray[np.float_] = DEFAULT_EARTH_POS,
) -> NDArray[np.float_]:
    """Tabulates the component densities for a meshgrid."""

    if not isinstance(model, InterplanetaryDustModel):
        model = model_registry.get_model(model)

    if not isinstance(grid, np.ndarray):
        grid = np.asarray(grid)

    x0_cloud = model.comps[CompLabel.CLOUD].x_0
    earth_coords = np.reshape(earth_coords, (3, 1, 1, 1))

    density_grid = np.zeros((model.ncomps, *grid.shape[1:]))
    for idx, comp in enumerate(model.comps.values()):
        comp.X_0 = np.reshape(comp.X_0, (3, 1, 1, 1))
        density_grid[idx] = comp.compute_density(
            X_helio=grid,
            X_earth=earth_coords,
            X_0_cloud=x0_cloud,
        )
        comp.X_0 = np.reshape(comp.X_0, (3, 1))

    return density_grid
