from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from zodipy._ipd_dens_funcs import PartialComputeDensityFunc
from zodipy._ipd_model import RRM, InterplanetaryDustModel, Kelsall
from zodipy._source_funcs import (
    get_bandpass_integrated_blackbody_emission,
    get_dust_grain_temperature,
    get_phase_function,
    get_scattering_angle,
)

"""Returns the zodiacal emission at a step along all lines of sight."""
GetEmissionAtStepCallable = Callable[..., npt.NDArray[np.float64]]


def get_emission_at_step_kelsall(
    r: npt.NDArray[np.float64],
    start: np.float64,
    stop: npt.NDArray[np.float64],
    gauss_quad_degree: int,
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    density_partials: tuple[PartialComputeDensityFunc],
    freq: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    T_0: float,
    delta: float,
    emissivities: npt.NDArray[np.float64],
    albedos: npt.NDArray[np.float64],
    phase_coefficients: tuple[float, ...],
    solar_irradiance: np.float64,
) -> npt.NDArray[np.float64]:
    """Kelsall uses common line of sight grid from obs to 5.2 AU."""

    # Convert the quadrature range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = get_bandpass_integrated_blackbody_emission(
        freq=freq,
        weights=weights,
        T=temperature,
    )

    emission = np.zeros((len(density_partials), stop.size, gauss_quad_degree))
    density = np.zeros_like(emission)
    for idx, (get_density_func, albedo, emissivity) in enumerate(
        zip(density_partials, albedos, emissivities)
    ):
        density[idx] = get_density_func(X_helio)
        emission[idx] = (1 - albedo) * (emissivity * blackbody_emission)

    if any(albedo != 0 for albedo in albedos):
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        for idx, albedo in enumerate(albedos):
            emission[idx] += albedo * solar_flux * phase_function

    return emission * density


def get_emission_at_step_rrm(
    r: npt.NDArray[np.float64],
    start: tuple[npt.NDArray[np.float64]],
    stop: tuple[npt.NDArray[np.float64]],
    gauss_quad_degree: int,
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    density_partials: tuple[PartialComputeDensityFunc],
    freq: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    T_0: tuple[float, ...],
    delta: tuple[float, ...],
    calibration: np.float64,
) -> npt.NDArray[np.float64]:
    """RRM has component specific line of sight grids."""

    emission = np.zeros((len(density_partials), stop[0].size, gauss_quad_degree))
    for idx, (
        get_density_func,
        start_comp,
        stop_comp,
        T_0_comp,
        delta_comp,
    ) in enumerate(zip(density_partials, start, stop, T_0, delta)):
        # Convert the quadrature range from [-1, 1] to the true ecliptic positions
        R_los = ((stop_comp - start_comp) / 2) * r + (stop_comp + start_comp) / 2

        X_los = R_los * u_los
        X_helio = X_los + X_obs
        R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

        temperature = get_dust_grain_temperature(R_helio, T_0_comp, delta_comp)
        blackbody_emission = get_bandpass_integrated_blackbody_emission(
            freq=freq,
            weights=weights,
            T=temperature,
        )
        emission[idx] = blackbody_emission * get_density_func(X_helio)

    return emission * calibration


EMISSION_MAPPING: dict[type[InterplanetaryDustModel], GetEmissionAtStepCallable] = {
    Kelsall: get_emission_at_step_kelsall,
    RRM: get_emission_at_step_rrm,
}
