import random

import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers

from zodipy import model_registry, tabulate_density
from zodipy.zodipy import Model

from ._strategies import zodipy_models


@given(
    zodipy_models(model="DIRBE"),
    integers(min_value=10, max_value=200),
    floats(min_value=2, max_value=10),
    floats(min_value=2, max_value=10),
    floats(min_value=2, max_value=10),
)
@settings(max_examples=10, deadline=None)
def test_tabulated_density(
    model: Model,
    n_grid_points: int,
    x_boundary: float,
    y_boundary: float,
    z_boundary: float,
) -> None:
    """Tests that the tabulated density is a 3D array of the correct shape."""
    x = np.linspace(-x_boundary, -x_boundary, n_grid_points)
    y = np.linspace(-y_boundary, y_boundary, n_grid_points)
    z = np.linspace(-z_boundary, z_boundary, n_grid_points)

    grid_regular = np.meshgrid(x, y, z)
    grid_array = np.asarray(grid_regular)
    grid = random.choice([grid_array, grid_regular])

    assert tabulate_density(grid, model="DIRBE").shape == (
        len(model._interplanetary_dust_model.comps),
        n_grid_points,
        n_grid_points,
        n_grid_points,
    )


def test_tabulated_density_str_model() -> None:
    """Tests that the tabulated density runs."""
    x = np.linspace(-5, -5, 100)
    y = np.linspace(-5, 5, 100)
    z = np.linspace(-2, 2, 100)

    grid = np.meshgrid(x, y, z)

    model = random.choice(model_registry.models)

    tabulate_density(grid, model=model)
