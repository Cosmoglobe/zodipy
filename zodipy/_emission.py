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


def get_emission_step(
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
    albedo: float,
    phase_coefficients: tuple[float, float, float],
) -> NDArray[np.floating]:
    """Returns the Zodiacal emission at a step along a line of sight.

    This function computes equation (1) in Kelsall et al. (1998) along a step in ds.

    Parameters
    ----------
    r
        Radial distance along the line-of-sight [AU / 1 AU].
    frequency
        Frequency at which to evaluate the brightness integral [GHz].
    observer_position
        The heliocentric ecliptic cartesian position of the observer [AU / 1 AU].
    earth_position
        The heliocentric ecliptic cartesian position of the Earth [AU / 1 AU].
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors for each pointing [AU / 1 AU].
    component
        Interplanetary Dust component.
    T_0
        Interplanetary temperature at 1 AU.
    delta
        Interplanetary temperature power law parameter.
    emissivity
        Frequency and component dependant emissivity factor representing the 
        deviation of the emisstion from a black body.
    albedo
        Frequency and component dependant albedo factor representing the
        probability of scatterin.
    phase_coefficient
        Frequency dependant parameters representing the distribution of scattered
        emission.

    Returns
    -------
        The Zodiacal emission from an Interplanetary Dust component at a step 
        along line of sights in units of W / Hz / m^2 / sr.
    """

    r_vec = r * unit_vectors
    X_helio = r_vec + observer_position
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    density = component.compute_density(X_helio=X_helio, X_earth=earth_position)
    interplanetary_temperature = get_interplanetary_temperature(R_helio, T_0, delta)
    blackbody_emission = get_blackbody_emission_nu(
        frequency, interplanetary_temperature
    )

    if albedo > 0:
        emission = (1 - albedo) * (emissivity * blackbody_emission)
        scattering_angle = np.arccos(np.sum(r_vec * X_helio, axis=0) / (r * R_helio))
        solar_flux = get_solar_flux(R_helio, frequency)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)
        emission += albedo * solar_flux * phase_function

    else:
        emission = emissivity * blackbody_emission

    return emission * density
