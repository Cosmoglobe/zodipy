from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from zodipy._ipd_dens_funcs import ComponentDensityFn
from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipy._source_funcs import (
    get_dust_grain_temperature,
    get_phase_function,
    get_scattering_angle,
)

"""
Function that returns the zodiacal emission at a step along all lines of sight given 
a zodiacal model.
"""
GetCompEmissionAtStepFn = Callable[..., npt.NDArray[np.float64]]


def kelsall(
    r: npt.NDArray[np.float64],
    start: np.float64,
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    get_density_function: ComponentDensityFn,
    T_0: float,
    delta: float,
    emissivity: np.float64,
    albedo: np.float64,
    phase_coefficients: tuple[float, ...],
    solar_irradiance: np.float64,
    bp_interpolation_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)
    emission = (1 - albedo) * (emissivity * blackbody_emission)

    if albedo != 0:
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        emission += albedo * solar_flux * phase_function

    return emission * get_density_function(X_helio)


def rrm(
    r: npt.NDArray[np.float64],
    start: npt.NDArray[np.float64],
    stop: npt.NDArray[np.float64],
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    get_density_function: ComponentDensityFn,
    T_0: float,
    delta: float,
    calibration: np.float64,
    bp_interpolation_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """RRM has component specific line of sight grids."""
    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2
    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = np.interp(temperature, *bp_interpolation_table)

    return blackbody_emission * get_density_function(X_helio) * calibration


EMISSION_MAPPING: dict[type[InterplanetaryDustModel], GetCompEmissionAtStepFn] = {
    Kelsall: kelsall,
    RRM: rrm,
}
