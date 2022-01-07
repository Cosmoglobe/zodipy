from typing import Dict

import healpy as hp
import numpy as np

from zodipy._labels import Label
from zodipy._model import IPDModel
from zodipy._line_of_sight import trapezoidal


def instantaneous_emission(
    nside: int,
    freq: float,
    model: IPDModel,
    line_of_sights: Dict[Label, np.ndarray],
    observer_positions: np.ndarray,
    earth_positions: np.ndarray,
    coord_out: str,
) -> np.ndarray:
    """Returns the simulated instantaneous emission.

    Parameters
    ----------
    nside
        HEALPIX map resolution parameter.
    freq
        Frequency at which to evaluate the Zodiacal emission [Hz].
    components
        List of Zodiacal Components for which to simulate the emission.
    source_parameters
        Parmeters for the evaluating the source function.
    observer_positions
        A sequence of (or a single) heliocentric cartesian positions of the 
        observer.
    earth_positions
        A sequence of (or a single) heliocentric cartesian positions of the 
        Earth.
    coord_out
        Coordinate frame of the output map.

    Returns
    -------
    emission
        Simulated (mean) instantaneous Zodiacal emission [MJy/sr].
    """


    npix = hp.nside2npix(nside)
    unit_vectors = _get_unit_vectors(
        nside=nside,
        pixels=np.arange(npix),
        coord_out=coord_out,
    )
    emission = np.zeros((len(model.components), npix))
    
    for observer_position, earth_position in zip(
        observer_positions,
        earth_positions,
    ):
        for idx, (label, component) in enumerate(model.components.items()):
            # source_function = get_source_function(model=model, freq=freq)
            integrated_comp_emission = trapezoidal(
                freq=freq,
                radial_distances=line_of_sights[label],
                get_density=component.get_density,
                observer_position=observer_position,
                earth_position=earth_position,
                unit_vectors=unit_vectors,
                T_0=model.interplanetary_temperature,
                delta=model.delta,
                source_function=1,
            )

            emission[idx] += integrated_comp_emission

    return emission * 1e20


def time_ordered_emission(
    nside: int,
    freq: float,
    model: IPDModel,
    line_of_sights: Dict[Label, np.ndarray],
    observer_position: np.ndarray,
    earth_position: np.ndarray,
    pixel_chunk: np.ndarray,
    bin: bool,
    coord_out: str,
) -> np.ndarray:
    """Simulates and returns the Zodiacal emission timestream.

    Parameters
    ----------
    nside
        HEALPIX map resolution parameter.
    freq
        Frequency at which to evaluate the Zodiacal emission [Hz].
    model
        Interplanetary dust model.
    observer_position
        Heliocentric cartesian position of the observer.
    earth_position
        Heliocentric cartesian position of the Earth.
    pixel_chunk
        An array representing a chunk of pixels where we assume the observer 
        and Earth position to be constant.
    bin
        If True, the time-ordered sequence of emission per pixel is binned 
        into a HEALPIX map. Defaults to False.
    coord_out
        Coordinate frame of the output map.

    Returns
    -------
        Simulated timestream of Zodiacal emission [MJy/sr] (optionally 
        binned into a HEALPIX map).
    """

    if bin:
        pixels, counts = np.unique(pixel_chunk, return_counts=True)
        unit_vectors = _get_unit_vectors(
            nside=nside,
            pixels=pixels,
            coord_out=coord_out,
        )
        emission = np.zeros((len(model.components), hp.nside2npix(nside)))

        for idx, (label, component) in enumerate(model.components.items()):
            integrated_comp_emission = trapezoidal(
                freq=freq,
                radial_distances=line_of_sights[label],
                get_density=component.get_density,
                observer_position=observer_position,
                earth_position=earth_position,
                unit_vectors=unit_vectors,
                T_0=model.interplanetary_temperature,
                delta=model.delta,
                source_function=1,
            )
            emission[idx, pixels] = integrated_comp_emission

        emission[:, pixels] *= counts

        return emission * 1e20

    pixels, indicies = np.unique(pixel_chunk, return_inverse=True)
    unit_vectors = _get_unit_vectors(
        nside=nside,
        pixels=pixels,
        coord_out=coord_out,
    )
    time_stream = np.zeros((len(model.components), len(pixel_chunk)))

    for idx, (label, component) in enumerate(model.components.items()):
        integrated_comp_emission = trapezoidal(
            freq=freq,
            radial_distances=line_of_sights[label],
            get_density=component.get_density,
            observer_position=observer_position,
            earth_position=earth_position,
            unit_vectors=unit_vectors,
            T_0=model.interplanetary_temperature,
            delta=model.delta,
            source_function=1,
        )

        time_stream[idx] = integrated_comp_emission[indicies]

    return time_stream * 1e20


def _get_unit_vectors(nside: int, pixels: np.ndarray, coord_out: str) -> np.ndarray:
    """Returns the unit vectors of a HEALPIX map given a requested output coordinate system.

    Since the Interplanetary Dust Model is evaluated in Ecliptic coordinates,
    we need to rotate any unit vectors defined in another coordinate frame to
    ecliptic before evaluating the model.
    """

    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))
    if coord_out != "E":
        unit_vectors = hp.rotator.Rotator(coord=[coord_out, "E"])(unit_vectors)

    return np.asarray(unit_vectors)