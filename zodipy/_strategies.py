from typing import List

import numpy as np
import healpy as hp

from zodipy._components import Component
from zodipy._integration import line_of_sight_integrate


def instantaneous_emission(
    nside: int,
    freq: float,
    components: List[Component],
    emissivities: List[float],
    line_of_sights: List[np.ndarray],
    observer_coords: np.ndarray,
    earth_coords: np.ndarray,
) -> np.ndarray:
    """Returns the instantaneous emission

    The emission is that seen by an observer at an instant at a single time
    or as the average of multiple times.

    Parameters:
    -----------
    nside
        HEALPIX map resolution parameter.
    freq:
        Frequency of the observer in GHz.
    components:
        Dictionary containing the Zodiacal Component that is used to evaluate
        the emission.
    emissivities:
        Sequency of emissivities, one for each component, corresponding to the
        frequency of `freq`.
    observer_coords
    """

    npix = hp.nside2npix(nside)
    unit_vectors = np.asarray(hp.pix2vec(nside, np.arange(npix)))

    emission = np.zeros((len(components), npix))
    for obs_pos, earth_pos in zip(
        observer_coords,
        earth_coords,
    ):
        for idx, component in enumerate(components):
            integrated_comp_emission = line_of_sight_integrate(
                line_of_sight=line_of_sights[idx],
                get_emission_func=component.get_emission,
                observer_coordinates=obs_pos,
                earth_coordinates=earth_pos,
                unit_vectors=unit_vectors,
                freq=freq,
            )

            emission[idx] += integrated_comp_emission * emissivities[idx]

    return emission * 1e20


def time_ordered_emission(
    pixel_chunk: np.ndarray,
    freq: float,
    nside: int,
    components: List[Component],
    emissivities: List[float],
    line_of_sights: List[np.ndarray],
    observer_coordinates: np.ndarray,
    earth_coordinates: np.ndarray,
    det=1
) -> np.ndarray:
    """Computes the Zodiacal Emission in the timestream given a scanning strategy."""

    pixels, indicies = np.unique(pixel_chunk, return_inverse=True)
    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))
    time_stream = np.zeros((len(components), len(pixel_chunk)))

    for idx, component in enumerate(components):
        integrated_comp_emission = line_of_sight_integrate(
            line_of_sight=line_of_sights[idx],
            get_emission_func=component.get_emission,
            observer_coordinates=observer_coordinates,
            earth_coordinates=earth_coordinates,
            unit_vectors=unit_vectors,
            freq=freq,
        )

        time_stream[idx] += integrated_comp_emission[indicies] * emissivities[idx]

    return time_stream * 1e20
