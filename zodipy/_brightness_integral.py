from typing import List, Tuple

import astropy.units as u
import numpy as np

from zodipy._model import InterplanetaryDustModel
import zodipy._source_functions as source_funcs
from zodipy._dirbe_bandpass import get_color_correction
from zodipy._labels import Label


def brightness_integral(
    model: InterplanetaryDustModel,
    component_label: Label,
    freq: float,
    radial_distances: np.ndarray,
    observer_position: np.ndarray,
    earth_position: np.ndarray,
    unit_vectors: np.ndarray,
) -> np.ndarray:
    """Returns the integrated Zodiacal emission for a component using the
    Trapezoidal method.

    Parameters
    ----------
    model
        Interplanetary Dust Model.
    component_label
        Component to evaluate.
    freq
        Frequency [GHz] at which to evaluate to Zodiacal Emission.
    radial_distances
        Array of discrete radial distances from the observer [AU].
    observer_position
        The heliocentric ecliptic cartesian position of the observer.
    earth_position
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
    component = model.get_initialized_component(component_label)
    emissivity, albedo, phase_coefficients = _get_interpolated_source_parameters(
        freq=freq * u.GHz,
        model=model,
        component=component_label,
    )
    T_0 = model.source_parameters["T_0"]
    delta = model.source_parameters["delta"]

    r = radial_distances[0]
    r_u = r * unit_vectors
    observer_position = np.expand_dims(observer_position, axis=1)
    X_helio = r_u + observer_position
    R_helio = np.linalg.norm(X_helio, axis=0)

    scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r * R_helio))

    solar_flux = source_funcs.blackbody_emission(5778, freq)
    phase_function = source_funcs.phase_function(
        Theta=scattering_angle, C=phase_coefficients
    )
    T = source_funcs.interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
    # color_correction = get_color_correction(T=T, freq=freq)

    B_nu = source_funcs.blackbody_emission(T=T, freq=freq)

    density = component.get_density(
        pixel_positions=X_helio, earth_position=earth_position
    )
    emission_previous = density * (
        albedo * solar_flux * phase_function
        + (1 - albedo) * (emissivity * B_nu)# * color_correction)
    )

    diff_radial_distances = np.diff(radial_distances)

    for r, dr in zip(radial_distances[1:], diff_radial_distances):
        X_helio = (r_u := r * unit_vectors) + observer_position
        R_helio = np.linalg.norm(X_helio, axis=0)
        scattering_angle = np.arccos(np.sum(r_u * X_helio, axis=0) / (r * R_helio))

        phase_function = source_funcs.phase_function(
            Theta=scattering_angle, C=phase_coefficients
        )
        T = source_funcs.interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
        # color_correction = get_color_correction(T=T, freq=freq)
        B_nu = source_funcs.blackbody_emission(T=T, freq=freq)

        density = component.get_density(
            pixel_positions=X_helio, earth_position=earth_position
        )
        emission_current = density * (
            albedo * solar_flux * phase_function
            + (1 - albedo) * (emissivity * B_nu)# * color_correction)
        )

        integrated_emission += (emission_previous + emission_current) * dr / 2
        emission_previous = emission_current

    return integrated_emission


def _get_interpolated_source_parameters(
    freq: u.Quantity, model: InterplanetaryDustModel, component: Label
) -> Tuple[float, float, List[float]]:
    """Returns interpolated source parameters given a frequency and a component."""

    emissivity_spectrum = model.source_component_parameters["emissivities"]["spectrum"]
    emissivity = np.interp(
        freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
        emissivity_spectrum,
        model.source_component_parameters["emissivities"][component],
    )

    albedo_spectrum = model.source_component_parameters["albedos"]["spectrum"]
    albedo = np.interp(
        freq.to(albedo_spectrum.unit, equivalencies=u.spectral()),
        albedo_spectrum,
        model.source_component_parameters["albedos"][component],
    )

    phase_spectrum = model.source_parameters["phase"]["spectrum"]

    phase_coefficients = [
        np.interp(
            freq.to(phase_spectrum.unit, equivalencies=u.spectral()),
            phase_spectrum,
            phase_coeff,
        )
        for phase_coeff in model.source_parameters["phase"]["coefficients"]
    ]

    return emissivity, albedo, phase_coefficients
