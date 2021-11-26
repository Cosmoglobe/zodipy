from typing import Protocol

import numpy as np


class ZodiacalComponentEmissionFunc(Protocol):
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
    line_of_sight: np.ndarray,
    get_emission_func: ZodiacalComponentEmissionFunc,
    observer_position: np.ndarray,
    earth_position: np.ndarray,
    unit_vectors: np.ndarray,
    freq: float,
) -> np.ndarray:
    """Integrates the emission for a component using the trapezoidal method.

    Parameters
    ----------
    line_of_sight
        Line-of-sight array.
    get_emission_func
        The `get_emission` function of the component.
    observer_position
        The heliocentric position of the observer.
    earth_position
        The heliocentric position of the Earth.
    unit_vectors
        Heliocentric Unit vectors pointing to each pixel in the HEALPIX map.
    freq
        Frequency at which to evaluate to Zodiacal Emission.

    Returns
    -------
    integrated_emission
        The line-of-sight integrated emission of the shells around an observer
        for a single Zodiacal Component.
    """

    emission_previous = get_emission_func(
        distance_to_shell=line_of_sight[0],
        observer_position=observer_position,
        earth_position=earth_position,
        unit_vectors=unit_vectors,
        freq=freq,
    )
    dR = np.diff(line_of_sight)

    integrated_emission = np.zeros(unit_vectors.shape[-1])
    for r, dr in zip(line_of_sight, dR):
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
