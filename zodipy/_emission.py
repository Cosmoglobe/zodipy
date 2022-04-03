from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zodipy._component import Component
from zodipy._source import (
    get_blackbody_emission_nu,
    get_interplanetary_temperature,
    get_phase_function,
    get_solar_flux,
)


def get_step_emission(
    r: float | NDArray[np.floating],
    *,
    frequency: float,
    observer_position: NDArray[np.floating],
    earth_position: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
    component: Component,
    T_0: float,
    delta: float,
    emissivity: float,
    albedo: float | None,
    phase_coefficients: tuple[float, float, float] | None,
    colorcorr_table: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Returns the Zodiacal emission at a step along the line-of-sight.

    This function computes equation (1) in K98 along a step in ds.

    Parameters
    ----------
    r
        Radial distance along the line-of-sight [AU / 1 AU].
    freq
        Frequency at which to evaluate the brightness integral [GHz].
    observer_pos
        The heliocentric ecliptic cartesian position of the observer [AU / 1 AU].
    earth_pos
        The heliocentric ecliptic cartesian position of the Earth [AU / 1 AU].
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each
        position in space we that we consider [AU / 1 AU].
    comp
        Zodiacal component class.
    cloud_offset
        Heliocentric ecliptic offset for the Zodiacal Cloud component in the
        model.
    source_params
        Dictionary containing various model and interpolated spectral
        parameters required for the evaluation of the brightness integral.

    Returns
    -------
    emission
        The Zodiacal emission at a step along the line-of-sight
        [W / Hz / m^2 / sr].
    """

    r_vec = r * unit_vectors
    X_helio = r_vec + observer_position
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    density = component.compute_density(X_helio=X_helio, X_earth=earth_position)
    interplanetary_temperature = get_interplanetary_temperature(R_helio, T_0, delta)
    blackbody_emission = get_blackbody_emission_nu(
        frequency, interplanetary_temperature
    )

    if albedo is not None and phase_coefficients is not None and albedo > 0:
        emission = (1 - albedo) * (emissivity * blackbody_emission)

        if colorcorr_table is not None:
            emission *= np.interp(interplanetary_temperature, *colorcorr_table)

        scattering_angle = np.arccos(np.sum(r_vec * X_helio, axis=0) / (r * R_helio))
        solar_flux = get_solar_flux(R_helio, frequency)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)
        emission += albedo * solar_flux * phase_function

    else:
        emission = emissivity * blackbody_emission

        if colorcorr_table is not None:
            emission *= np.interp(interplanetary_temperature, *colorcorr_table)

    return emission * density
