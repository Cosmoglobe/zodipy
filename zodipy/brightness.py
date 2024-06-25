from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from zodipy.blackbody import get_dust_grain_temperature
from zodipy.scattering import get_phase_function, get_scattering_angle

if TYPE_CHECKING:
    from zodipy.number_density import ComponentNumberDensityCallable

"""
Function that return the zodiacal emission at a step along all lines of sight given
a zodiacal model.
"""
BrightnessAtStepCallable = Callable[..., npt.NDArray[np.float64]]


def kelsall_brightness_at_step(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    quad_root_0: float,
    quad_root_n: float,
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    bp_interpolation_table: npt.NDArray[np.float64],
    get_density_function: ComponentNumberDensityCallable,
    T_0: float,
    delta: float,
    emissivity: np.float64,
    albedo: np.float64,
    C1: np.float64,
    C2: np.float64,
    C3: np.float64,
    solar_irradiance: np.float64,
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""
    # Convert the quadrature range from [0, inf] to the true ecliptic positions
    a, b = start, stop
    i, j = quad_root_0, quad_root_n
    R_los = ((b - a) / (j - i)) * r + (j * a - i * b) / (j - i)

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)
    emission = (1 - albedo) * (emissivity * blackbody_emission)

    if albedo != 0:
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, C1, C2, C3)
        emission += albedo * solar_flux * phase_function

    return emission * get_density_function(X_helio)


def rrm_brightness_at_step(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    quad_root_0: float,
    quad_root_n: float,
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    bp_interpolation_table: npt.NDArray[np.float64],
    get_density_function: ComponentNumberDensityCallable,
    T_0: float,
    delta: float,
    calibration: np.float64,
) -> npt.NDArray[np.float64]:
    """RRM is implented with component specific line-of-sight grids."""
    # Convert the quadrature range from [0, inf] to the true ecliptic positions
    a, b = start, stop
    i, j = quad_root_0, quad_root_n
    R_los = ((b - a) / (j - i)) * r + (j * a - i * b) / (j - i)

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)

    return blackbody_emission * get_density_function(X_helio) * calibration
