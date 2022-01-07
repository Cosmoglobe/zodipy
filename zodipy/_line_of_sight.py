from typing import Protocol

import numpy as np

from zodipy._functions import interplanetary_temperature, blackbody_emission


class GetDensityFunction(Protocol):
    def __call__(
        self, pixel_positions: np.ndarray, earth_position: np.ndarray
    ) -> np.ndarray:
        """Interface of the `get_emission` function of a Zodiacal Component."""


def trapezoidal(
    freq: float,
    radial_distances: np.ndarray,
    get_density_func: GetDensityFunction,
    observer_position: np.ndarray,
    earth_position: np.ndarray,
    unit_vectors: np.ndarray,
    T_0: float,
    delta: float,
    source_function: float,
) -> np.ndarray:
    """Returns the integrated Zodiacal emission for a component (Trapezoidal).

    Parameters
    ----------
    freq
        Frequency at which to evaluate to Zodiacal Emission.
    radial_distances
        Array of discrete radial distances from the observer [AU].
    get_density_func
        The `get_emission` function of the component.
    observer_position
        The heliocentric ecliptic cartesian position of the observer.
    earth_position
        The heliocentric ecliptic cartesian position of the Earth.
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each
        position in space we that we consider.
    T_0
        Interplanetary temperature at 1 AU.
    delta
        Power law exponent for the interplanetary temperature.
    source_function
        The source function without the blackbody term as defined in K98.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated emission of the shells around an observer
        for a Zodiacal Component.
    """

    observer_position = np.expand_dims(observer_position, axis=1)

    X_helio = radial_distances[0] * unit_vectors + observer_position
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)
    T = interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
    B_nu = blackbody_emission(T=T, nu=freq)
    density = get_density_func(pixel_positions=X_helio, earth_position=earth_position)

    emission_previous = density * B_nu * source_function

    dR = np.diff(radial_distances)
    integrated_emission = np.zeros(unit_vectors.shape[-1])
    for r, dr in zip(radial_distances, dR):
        X_helio = r * unit_vectors + observer_position
        R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)
        T = interplanetary_temperature(R=R_helio, T_0=T_0, delta=delta)
        B_nu = blackbody_emission(T=T, nu=freq)
        density = get_density_func(
            pixel_positions=X_helio, earth_position=earth_position
        )
        emission_current = density * B_nu * source_function
        integrated_emission += (emission_previous + emission_current) * dr / 2
        emission_previous = emission_current

    return integrated_emission
