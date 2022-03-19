from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from zodipy._source_funcs import phase_function
from zodipy._components import Component
from zodipy._interp import (
    interp_blackbody_emission_nu,
    interp_interplanetary_temperature,
    interp_solar_flux,
)


def trapezoidal(
    comp: Component,
    freq: float,
    line_of_sight: NDArray[np.floating],
    observer_pos: NDArray[np.floating],
    earth_pos: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
    cloud_offset: NDArray[np.floating],
    T_0: float,
    delta: float,
    emissivity: float,
    albedo: float,
    phase_coeffs: Sequence[float],
    colorcorr_table: Optional[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method.

    Parameters
    ----------
    comp
        Zodiacal component to evaluate.
    freq
        Frequency at which to evaluate the brightness integral [GHz].
    line_of_sight
        Array of discrete radial distances from the observer [AU / 1 AU].
    observer_pos
        The heliocentric ecliptic cartesian position of the observer [AU / 1 AU].
    earth_pos
        The heliocentric ecliptic cartesian position of the Earth [AU / 1 AU].
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each
        position in space we that we consider [AU / 1 AU].
    cloud_offset
        Heliocentric ecliptic offset for the Zodiacal Cloud component.
    source_params
        Dictionary containing various model and interpolated spectral
        parameters required for the evaluation of the brightness integral.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated emission for the observer [W / Hz / m^2 / sr].
    """

    integrated_emission = np.zeros(unit_vectors.shape[-1])

    emission_previous = get_step_emission(
        comp=comp,
        freq=freq,
        r=line_of_sight[0],
        observer_pos=observer_pos,
        earth_pos=earth_pos,
        unit_vectors=unit_vectors,
        cloud_offset=cloud_offset,
        T_0=T_0,
        delta=delta,
        emissivity=emissivity,
        albedo=albedo,
        phase_coeffs=phase_coeffs,
        colorcorr_table=colorcorr_table,
    )

    for r, dr in zip(line_of_sight[1:], np.diff(line_of_sight)):
        emission_current = get_step_emission(
            comp=comp,
            freq=freq,
            r=r,
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            unit_vectors=unit_vectors,
            cloud_offset=cloud_offset,
            T_0=T_0,
            delta=delta,
            emissivity=emissivity,
            albedo=albedo,
            phase_coeffs=phase_coeffs,
            colorcorr_table=colorcorr_table,
        )

        integrated_emission += (emission_previous + emission_current) * (dr / 2)

        emission_previous = emission_current

    return integrated_emission


def get_step_emission(
    comp: Component,
    freq: float,
    r: NDArray[np.floating],
    observer_pos: NDArray[np.floating],
    earth_pos: NDArray[np.floating],
    unit_vectors: NDArray[np.floating],
    cloud_offset: NDArray[np.floating],
    T_0: float,
    delta: float,
    emissivity: float,
    albedo: float,
    phase_coeffs: Sequence[float],
    colorcorr_table: Optional[NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Returns the Zodiacal emission at a step along the line-of-sight.

    Parameters
    ----------
    comp
        Zodiacal component to evaluate.
    freq
        Frequency at which to evaluate the brightness integral [GHz].
    r
        Radial distance along the line-of-sight [AU / 1 AU].
    observer_pos
        The heliocentric ecliptic cartesian position of the observer [AU / 1 AU].
    earth_pos
        The heliocentric ecliptic cartesian position of the Earth [AU / 1 AU].
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each
        position in space we that we consider [AU / 1 AU].
    cloud_offset
        Heliocentric ecliptic offset for the Zodiacal Cloud component.
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

    X_helio = r_vec + observer_pos
    R_helio = np.sqrt(X_helio[0]**2 + X_helio[1]**2 + X_helio[2]**2)

    density = comp.compute_density(
        X_helio=X_helio,
        X_earth=earth_pos,
        X_0_cloud=cloud_offset,
    )

    T = interp_interplanetary_temperature(
        R=R_helio,
        T_0=T_0,
        delta=delta,
    )
    B_nu = interp_blackbody_emission_nu(T=T, freq=freq)

    emission = (1 - albedo) * (emissivity * B_nu)

    if colorcorr_table is not None:
        emission *= np.interp(T, *colorcorr_table)

    if albedo > 0:
        scattering_angle = np.arccos(np.sum(r_vec * X_helio, axis=0) / (r * R_helio))
        solar_flux = interp_solar_flux(R_helio, freq)
        phase = phase_function(scattering_angle, *phase_coeffs)
        emission += albedo * solar_flux * phase

    emission *= density

    return emission
