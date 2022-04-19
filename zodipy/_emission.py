from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._component import Component
from ._source_functions import (
    get_blackbody_emission_nu,
    get_interplanetary_temperature,
    get_phase_function,
    get_scattering_angle,
)


def get_emission_at_step(
    r: float | NDArray[np.floating],
    *,
    start: float,
    stop: float | NDArray[np.floating],
    X_obs: NDArray[np.floating],
    X_earth: NDArray[np.floating],
    u_los: NDArray[np.floating],
    component: Component,
    frequency: float,
    T_0: float,
    delta: float,
    emissivity: float,
    albedo: float,
    phase_coefficients: tuple[float, float, float],
    solar_irradiance: float,
) -> NDArray[np.floating]:
    """Returns the Zodiacal emission at a step along a line of sight.

    This function computes equation (1) in Kelsall et al. (1998) along a step in ds.

    Parameters
    ----------
    r
       Position along the normalized line of sight from [-1, 1] [AU].
    X_obs
        The heliocentric ecliptic cartesian position of the observer [AU].
    X_earth
        The heliocentric ecliptic cartesian position of the Earth [AU ].
    u_los
        Heliocentric ecliptic cartesian unit vectors for each pointing [AU].
    component
        Interplanetary Dust component.
    frequency
        Frequency at which to evaluate the brightness integral [GHz].
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
    solar_irradiance
        Solar irradiance (at 1 AU) at the frequency specified by 'frequency', given some
        Solar irradiance model [W / Hz / m^2 / sr].
    Returns
    -------
        The Zodiacal emission from an Interplanetary Dust component at a step
        along line of sights in units of W / Hz / m^2 / sr.
    """

    # Compute the true position along the line of sight from the substituted
    # limits of [-1, 1]
    R_los = ((stop - start) / 2) * r + (stop + start) / 2

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    density = component.compute_density(X_helio=X_helio, X_earth=X_earth)
    interplanetary_temperature = get_interplanetary_temperature(R_helio, T_0, delta)
    blackbody_emission = get_blackbody_emission_nu(
        frequency, interplanetary_temperature
    )

    emission = (1 - albedo) * (emissivity * blackbody_emission)

    if albedo > 0:
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        emission += albedo * solar_flux * phase_function

    return emission * density
