from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import numpy.typing as npt

from zodipy.blackbody import get_dust_grain_temperature
from zodipy.scattering import get_phase_function, get_scattering_angle

if TYPE_CHECKING:
    from zodipy.number_density import NumberDensityFunc

"""
Function that return the zodiacal emission at a step along all lines of sight given
a zodiacal model.
"""
BrightnessAtStepCallable = Callable[..., npt.NDArray[np.float64]]


def kelsall_brightness_at_step(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    bp_interpolation_table: npt.NDArray[np.float64],
    number_density_func: NumberDensityFunc,
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
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    # and back again at the end
    R_los = 0.5 * (stop - start) * r + 0.5 * (stop + start)

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

    return emission * number_density_func(X_helio) * 0.5 * (stop - start)


def rrm_brightness_at_step(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    bp_interpolation_table: npt.NDArray[np.float64],
    number_density_func: NumberDensityFunc,
    T_0: float,
    delta: float,
    calibration: np.float64,
) -> npt.NDArray[np.float64]:
    """RRM is implented with component specific line-of-sight grids."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    # and back again at the end
    R_los = 0.5 * (stop - start) * r + 0.5 * (stop + start)

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)

    return blackbody_emission * number_density_func(X_helio) * calibration * 0.5 * (stop - start)
