from typing import Dict, Optional

import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel
from zodipy._brightness_integral import brightness_integral

_SI_TO_MJY = 1e20


def instantaneous_emission(
    model: InterplanetaryDustModel,
    freq: float,
    nside: int,
    line_of_sights: Dict[Label, NDArray[np.float64]],
    observer_pos: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    color_table: Optional[NDArray[np.float64]],
    coord_out: str,
) -> NDArray[np.float64]:
    """Returns the simulated instantaneous emission.

    Parameters
    ----------
    nside
        HEALPIX map resolution parameter.
    freq
        Frequency at which to evaluate the Zodiacal emission [GHz].
    model
        Interplanetary dust model.
    line_of_sights
        Dictionary mapping radial line of sights from the observer to a
        Zodiacal component.
    observer_pos
        A sequence of (or a single) heliocentric cartesian positions of the
        observer.
    earth_pos
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
    unit_vectors = _get_rotated_unit_vectors(
        nside=nside,
        pixels=np.arange(npix),
        coord_out=coord_out,
    )
    emission = np.zeros((len(model.components), npix))
    for observer_position, earth_position in zip(observer_pos, earth_pos):
        for idx, label in enumerate(model.components):
            integrated_comp_emission = brightness_integral(
                model=model,
                component_label=label,
                freq=freq,
                radial_distances=line_of_sights[label],
                observer_pos=observer_position,
                earth_pos=earth_position,
                color_table=color_table,
                unit_vectors=unit_vectors,
            )

            emission[idx] += integrated_comp_emission

    return emission * _SI_TO_MJY


def time_ordered_emission(
    model: InterplanetaryDustModel,
    nside: int,
    freq: float,
    line_of_sights: Dict[Label, NDArray[np.float64]],
    observer_pos: NDArray[np.float64],
    earth_pos: NDArray[np.float64],
    pixel_chunk: NDArray[np.int64],
    color_table: Optional[NDArray[np.float64]],
    bin: bool,
    coord_out: str,
) -> NDArray[np.float64]:
    """Simulates and returns the Zodiacal emission timestream.

    Parameters
    ----------
    nside
        HEALPIX map resolution parameter.
    freq
        Frequency at which to evaluate the Zodiacal emission [Hz].
    model
        Interplanetary dust model.
    line_of_sights
        Dictionary mapping radial line of sights from the observer to a
        Zodiacal component.
    observer_pos
        Heliocentric cartesian position of the observer.
    earth_pos
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
        unit_vectors = _get_rotated_unit_vectors(
            nside=nside, pixels=pixels, coord_out=coord_out
        )
        emission = np.zeros((len(model.components), hp.nside2npix(nside)))

        for idx, label in enumerate(model.components):
            integrated_comp_emission = brightness_integral(
                model=model,
                component_label=label,
                freq=freq,
                radial_distances=line_of_sights[label],
                observer_pos=observer_pos,
                earth_pos=earth_pos,
                color_table=color_table,
                unit_vectors=unit_vectors,
            )
            emission[idx, pixels] = integrated_comp_emission

        emission[:, pixels] *= counts

        return emission * _SI_TO_MJY

    pixels, indicies = np.unique(pixel_chunk, return_inverse=True)
    unit_vectors = _get_rotated_unit_vectors(
        nside=nside, pixels=pixels, coord_out=coord_out
    )
    emission = np.zeros((len(model.components), len(pixel_chunk)))

    for idx, label in enumerate(model.components):
        integrated_comp_emission = brightness_integral(
            model=model,
            component_label=label,
            freq=freq,
            radial_distances=line_of_sights[label],
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            color_table=color_table,
            unit_vectors=unit_vectors,
        )

        emission[idx] = integrated_comp_emission[indicies]

    return emission * _SI_TO_MJY


def _get_rotated_unit_vectors(
    nside: int, pixels: NDArray[np.int64], coord_out: str
) -> NDArray[np.float64]:
    """Returns the unit vectors of a HEALPIX map given a requested output coordinate system.

    Since the Interplanetary Dust Model is evaluated in Ecliptic coordinates,
    we need to rotate any unit vectors defined in another coordinate frame to
    ecliptic before evaluating the model.
    """

    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))
    if coord_out != "E":
        unit_vectors = hp.rotator.Rotator(coord=[coord_out, "E"])(unit_vectors)

    return np.asarray(unit_vectors)
