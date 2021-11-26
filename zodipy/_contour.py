from math import sin, cos
from typing import Tuple, Sequence, Union

from zodipy.models import model_registry
from zodipy._component_labels import ComponentLabel
import numpy as np
import astropy.constants as const


def get_component_density_grid(
    component: str,
    model: str = "K98",
    earth_coords: Sequence[float] = (0, 1, 0),
    xy_lim: int = 5,
    z_lim: int = 1,
    n: int = 200,
    return_grid: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Returns a 3D grid of the components density.

    Parameters
    ----------
    component
        The component for which we grid the density.
    model
        The Interplanetary Dust Model that contains the component.
    xy_lim
        xy-limit in AU.
    z_lim
        z-limit in AU.
    n
        Number of grid points in each axis.
    return_grid
        If True, then the meshgrid is returned alongside the density_grid.

    Returns
    -------
    density_grid, XX, YY, ZZ
        3D grid of the density of the component.
    """

    ipd_model = model_registry.get_model(model)
    component_class = ipd_model.components[ComponentLabel(component)]

    x_helio = np.linspace(-xy_lim, xy_lim, n)
    y_helio = x_helio.copy()
    z_helio = np.linspace(-z_lim, z_lim, n)

    x_prime = x_helio - component_class.x_0
    y_prime = y_helio - component_class.y_0
    z_prime = z_helio - component_class.z_0

    XX_helio, YY_helio, ZZ_helio = np.meshgrid(x_helio, y_helio, z_helio)
    XX_prime, YY_prime, ZZ_prime = np.meshgrid(x_prime, y_prime, z_prime)

    R_prime = np.sqrt(XX_prime ** 2 + YY_prime ** 2 + ZZ_prime ** 2)
    Z_prime = (
        XX_prime * sin(component_class.Ω) * sin(component_class.i)
        - YY_prime * cos(component_class.Ω) * sin(component_class.i)
        + ZZ_prime * cos(component_class.i)
    )

    X_earth_prime = earth_coords - component_class.X_component[0]
    θ_prime = np.arctan2(YY_prime, ZZ_prime) - np.arctan2(
        X_earth_prime[1], X_earth_prime[0]
    )

    density_grid = component_class.get_density(
        R_prime=R_prime, Z_prime=Z_prime, θ_prime=θ_prime
    )

    # The densities in the model is given in units of 1 / AU
    density_grid *= const.au.value

    if return_grid:
        return density_grid, XX_helio, YY_helio, ZZ_helio

    return density_grid
