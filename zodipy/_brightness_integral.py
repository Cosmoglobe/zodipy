from typing import Any, Dict, Optional

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel
import zodipy._source_functions as source_funcs
from zodipy._components import Component


def trapezoidal(
    freq: float,
    model: InterplanetaryDustModel,
    component: Label,
    radial_distances: NDArray[np.float64],
    observer_pos: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    unit_vectors: NDArray[np.float64],
    color_table: Optional[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method.

    Parameters
    ----------
    freq
        Frequency at which to evaluate the brightness integral [GHz].
    model
        Interplanetary Dust Model.
    component
        Component (label) to evaluate.
    radial_distances
        Array of discrete radial distances from the observer [AU].
    observer_pos
        The heliocentric ecliptic cartesian position of the observer.
    earth_pos
        The heliocentric ecliptic cartesian position of the Earth.
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each
        position in space we that we consider.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated emission of the shells around an observer
        for a Zodiacal Component.
    """

    integrated_emission = np.zeros(unit_vectors.shape[-1])

    source_params = source_funcs.get_source_parameters(
        freq=freq,
        model=model,
        component=component,
    )

    emission_previous = get_emission(
        component=model.components[component],
        freq=freq,
        r=radial_distances[0],
        observer_pos=observer_pos,
        earth_pos=earth_pos,
        unit_vectors=unit_vectors,
        cloud_offset=source_params["cloud_offset"],
        color_table=color_table,
        albedo=source_params["albedo"],
        emissivity=source_params["emissivity"],
        phase_coeff=source_params["phase_coeffs"],
        T_0=source_params["T_0"],
        delta=source_params["delta"],
    )

    for r, dr in zip(radial_distances[1:], np.diff(radial_distances)):
        emission_current = get_emission(
            component=model.components[component],
            freq=freq,
            r=r,
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            unit_vectors=unit_vectors,
            cloud_offset=source_params["cloud_offset"],
            color_table=color_table,
            albedo=source_params["albedo"],
            emissivity=source_params["emissivity"],
            phase_coeff=source_params["phase_coeffs"],
            T_0=source_params["T_0"],
            delta=source_params["delta"],
        )
        integrated_emission += (emission_previous + emission_current) * (dr / 2)
        emission_previous = emission_current

    return integrated_emission


def get_emission(
    component: Component,
    freq: float,
    r: NDArray[np.float64],
    observer_pos: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    unit_vectors: NDArray[np.float64],
    cloud_offset: NDArray[np.float64],
    color_table: Optional[NDArray[np.float64]],
    albedo: float,
    emissivity: float,
    phase_coeff: Dict[str, Any],
    T_0: float,
    delta: float,
):
    """Returns the source term in the brightness integral as defined in K98."""

    r_u = r * unit_vectors
    observer_pos = np.expand_dims(observer_pos, axis=1)
    X_helio = r_u + observer_pos
    R_helio = np.linalg.norm(X_helio, axis=0)
    scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r * R_helio))

    solar_flux = source_funcs.solar_flux(R=R_helio, freq=freq) / r
    phase_function = source_funcs.phase_function(
        Theta=scattering_angle,
        coeffs=phase_coeff,
    )
    T = source_funcs.interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
    B_nu = source_funcs.blackbody_emission_nu(T=T, freq=freq)

    density = component.get_density(
        pixel_pos=X_helio,
        earth_pos=earth_pos,
        cloud_offset=cloud_offset,
    )

    emission = albedo * solar_flux * phase_function

    if color_table is not None:
        color_corr_factor = np.interp(T, color_table[0], color_table[1])
        emission += (1 - albedo) * (emissivity * B_nu * color_corr_factor)
    else:
        emission += (1 - albedo) * (emissivity * B_nu)

    emission *= density

    return emission
