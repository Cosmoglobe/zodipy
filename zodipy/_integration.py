from typing import Protocol

import numpy as np


class EmissionFunction(Protocol):
    def __call__(
        self,
        distance_to_shell: np.ndarray,
        observer_position: np.ndarray,
        earth_position: np.ndarray,
        unit_vectors: np.ndarray,
        freq: float,
    ) -> np.ndarray:
        """Interface of the `get_emission` function of a Zodiacal Component."""


def line_of_sight_integrate(
    radial_distances: np.ndarray,
    get_emission_func: EmissionFunction,
    observer_position: np.ndarray,
    earth_position: np.ndarray,
    unit_vectors: np.ndarray,
    freq: float,
) -> np.ndarray:
    """Returns the integrated Zodiacal emission for a component (Trapezoidal).

    Parameters
    ----------
    radial_distances
        Array of discrete radial distances from the observer [AU].
    get_emission_func
        The `get_emission` function of the component.
    observer_position
        The heliocentric ecliptic cartesian position of the observer.
    earth_position
        The heliocentric ecliptic cartesian position of the Earth.
    unit_vectors
        Heliocentric ecliptic cartesian unit vectors pointing to each 
        position in space we that we consider.    
    freq
        Frequency at which to evaluate to Zodiacal Emission.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated emission of the shells around an observer
        for a Zodiacal Component.
    """

    emission_previous = get_emission_func(
        distance_to_shell=radial_distances[0],
        observer_position=observer_position,
        earth_position=earth_position,
        unit_vectors=unit_vectors,
        freq=freq,
    )
    dR = np.diff(radial_distances)

    integrated_emission = np.zeros(unit_vectors.shape[-1])
    for r, dr in zip(radial_distances, dR):
        emission_current = get_emission_func(
            distance_to_shell=r,
            observer_position=observer_position,
            earth_position=earth_position,
            unit_vectors=unit_vectors,
            freq=freq,
        )
        integrated_emission += (emission_previous + emission_current) * dr / 2
        emission_previous = emission_current

    return integrated_emission
