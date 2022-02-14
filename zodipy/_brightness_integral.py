from typing import Any, Dict, Optional, Tuple

import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel
import zodipy._source_functions as source_funcs
from zodipy._components import Component

def trapezoidal(
    freq: Quantity[u.GHz],
    model: InterplanetaryDustModel,
    component: Tuple[Label, Component],
    radial_distances: Quantity[u.AU],
    observer_pos: Quantity[u.AU],
    earth_pos: Quantity[u.AU],
    unit_vectors: NDArray[np.float64],
    color_table: Optional[Tuple[Quantity[u.K], NDArray[np.float64]]],
) -> Quantity[u.MJy / u.sr]:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method.

    Parameters
    ----------
    freq
        Frequency at which to evaluate the brightness integral [GHz].
    model
        Interplanetary Dust Model.
    component
        Component to evaluate.
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

    component_label, component_cls = component

    # Extract model parameters
    T_0 = model.source_parameters["T_0"]
    delta = model.source_parameters["delta"]
    cloud_offset = model.components[component_label].X_0

    integrated_emission = Quantity(
        value=np.zeros(unit_vectors.shape[-1]),
        unit=u.W / u.Hz / u.m ** 2 / u.sr,
    )
    emissivity, albedo, phase_coeff = source_funcs.get_interpolated_source_parameters(
        freq=freq,
        model=model,
        component=component_label,
    )
    
    emission_previous = get_emission(
        component=component_cls,
        freq=freq,
        r=radial_distances[0],
        observer_pos=observer_pos,
        earth_pos=earth_pos,
        unit_vectors=unit_vectors,
        cloud_offset=cloud_offset,
        color_table=color_table,
        albedo=albedo,
        emissivity=emissivity,
        phase_coeff=phase_coeff,
        T_0=T_0,
        delta=delta,
    )

    for r, dr in zip(radial_distances[1:], np.diff(radial_distances)):
        emission_current = get_emission(
            component=component_cls,
            freq=freq,
            r=r,
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            unit_vectors=unit_vectors,
            cloud_offset=cloud_offset,
            color_table=color_table,
            albedo=albedo,
            emissivity=emissivity,
            phase_coeff=phase_coeff,
            T_0=T_0,
            delta=delta,
        )
        integrated_emission += (emission_previous + emission_current) * (dr / 2)
        emission_previous = emission_current

    return integrated_emission.to(u.MJy / u.sr)


def get_emission(
    component: Component,
    freq: Quantity[u.GHz],
    r: Quantity[u.AU],
    observer_pos: Quantity[u.AU],
    earth_pos: Quantity[u.AU],
    unit_vectors: NDArray[np.float64],
    cloud_offset: Quantity[u.dimensionless_unscaled],
    color_table: Optional[Tuple[Quantity[u.K], NDArray[np.float64]]],
    albedo: float,
    emissivity: float,
    phase_coeff: Dict[str, Any],
    T_0: Quantity[u.K],
    delta: float,
):
    """Returns the source term in the brightness integral as defined in K98."""

    r_u = r * unit_vectors
    observer_pos = np.expand_dims(observer_pos, axis=1)

    X_helio = r_u + observer_pos
    R_helio = np.linalg.norm(X_helio, axis=0)
    scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r * R_helio))

    solar_flux = source_funcs.solar_flux(R=R_helio, freq=freq)
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