from typing import Optional

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel
import zodipy._source_functions as source_funcs


def brightness_integral(
    freq: float,
    model: InterplanetaryDustModel,
    component_label: Label,
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
    component_label
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

    T_0 = model.source_parameters["T_0"]
    delta = model.source_parameters["delta"]
    cloud_x0 = model.component_parameters[Label.CLOUD]["x_0"]
    cloud_y0 = model.component_parameters[Label.CLOUD]["y_0"]
    cloud_z0 = model.component_parameters[Label.CLOUD]["z_0"]
    cloud_offset = np.expand_dims(np.array([cloud_x0, cloud_y0, cloud_z0]), axis=1)

    integrated_emission = np.zeros(unit_vectors.shape[-1])
    component = model.get_initialized_component(component_label)
    emissivity, albedo, phase_coeff = source_funcs.get_interpolated_source_parameters(
        freq=freq * u.GHz,
        model=model,
        component=component_label,
    )

    r_0 = radial_distances[0]
    r_u = r_0 * unit_vectors
    observer_pos = np.expand_dims(observer_pos, axis=1)
    X_helio = r_u + observer_pos
    R_helio = np.linalg.norm(X_helio, axis=0)
    scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r_0 * R_helio))

    solar_flux = source_funcs.blackbody_emission(5778, freq)
    phase_function = source_funcs.phase_function(Theta=scattering_angle, C=phase_coeff)
    T = source_funcs.interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
    B_nu = source_funcs.blackbody_emission(T=T, freq=freq)
    density = component.get_density(
        pixel_pos=X_helio,
        earth_pos=earth_pos,
        cloud_offset=cloud_offset,
    )

    emission_previous = albedo * solar_flux * phase_function
    if color_table is not None:
        color_corr_factor = np.interp(T, color_table[:, 0], color_table[:, 1])
        emission_previous += (1 - albedo) * (emissivity * B_nu * color_corr_factor)
    else:
        emission_previous += (1 - albedo) * (emissivity * B_nu)
    emission_previous *= density

    for r, dr in zip(radial_distances[1:], np.diff(radial_distances)):
        X_helio = (r_u := r * unit_vectors) + observer_pos
        R_helio = np.linalg.norm(X_helio, axis=0)

        scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r * R_helio))
        phase_function = source_funcs.phase_function(
            Theta=scattering_angle, C=phase_coeff
        )
        T = source_funcs.interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
        B_nu = source_funcs.blackbody_emission(T=T, freq=freq)

        density = component.get_density(
            pixel_pos=X_helio,
            earth_pos=earth_pos,
            cloud_offset=cloud_offset,
        )
        emission_current = albedo * solar_flux * phase_function
        if color_table is not None:
            color_corr_factor = np.interp(T, color_table[:, 0], color_table[:, 1])
            emission_current += (1 - albedo) * (emissivity * B_nu * color_corr_factor)
        else:
            emission_current += (1 - albedo) * (emissivity * B_nu)
        emission_current *= density

        integrated_emission += (emission_previous + emission_current) * (dr / 2)
        emission_previous = emission_current

    return integrated_emission
