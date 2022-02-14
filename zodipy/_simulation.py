from typing import Dict, Optional, Tuple

import astropy.units as u
from astropy.units import Quantity
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel
from zodipy._brightness_integral import trapezoidal


def instantaneous_emission(
    model: InterplanetaryDustModel,
    freq: Quantity[u.GHz],
    nside: int,
    line_of_sights: Dict[Label, Quantity[u.AU]],
    observer_pos: Quantity[u.AU],
    earth_pos: Quantity[u.AU],
    color_table: Optional[Tuple[Quantity[u.K], NDArray[np.float64]]],
    coord_out: str,
) -> Quantity[u.MJy / u.sr]:
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
    emission = Quantity(
        value=np.zeros((model.ncomps, npix)),
        unit=u.MJy / u.sr,
    )
    for observer_position, earth_position in zip(observer_pos, earth_pos):
        for idx, component in enumerate(model.components.items()):
            integrated_comp_emission = trapezoidal(
                model=model,
                component=component,
                freq=freq,
                radial_distances=line_of_sights[component[0]],
                observer_pos=observer_position,
                earth_pos=earth_position,
                color_table=color_table,
                unit_vectors=unit_vectors,
            )

            emission[idx] += integrated_comp_emission

    return emission


def time_ordered_emission(
    model: InterplanetaryDustModel,
    nside: int,
    freq: Quantity[u.GHz],
    line_of_sights: Dict[Label, Quantity[u.AU]],
    observer_pos: Quantity[u.AU],
    earth_pos: Quantity[u.AU],
    pixel_chunk: NDArray[np.int64],
    color_table: Optional[Tuple[Quantity[u.K], NDArray[np.float64]]],
    bin: bool,
    coord_out: str,
) -> Quantity[u.MJy / u.sr]:
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

    if earth_pos.ndim == 1:
        earth_pos = np.expand_dims(earth_pos, axis=1)

    if bin:
        pixels, counts = np.unique(pixel_chunk, return_counts=True)
        unit_vectors = _get_rotated_unit_vectors(
            nside=nside, pixels=pixels, coord_out=coord_out
        )
        emission = Quantity(
            value=np.zeros((model.ncomps, hp.nside2npix(nside))),
            unit=u.MJy / u.sr,
        )
        for idx, component in enumerate(model.components.items()):
            integrated_comp_emission = trapezoidal(
                model=model,
                component=component,
                freq=freq,
                radial_distances=line_of_sights[component[0]],
                observer_pos=observer_pos,
                earth_pos=earth_pos,
                color_table=color_table,
                unit_vectors=unit_vectors,
            )
            emission[idx, pixels] = integrated_comp_emission

        emission[:, pixels] *= counts

        return emission

    pixels, indicies = np.unique(pixel_chunk, return_inverse=True)
    unit_vectors = _get_rotated_unit_vectors(
        nside=nside, pixels=pixels, coord_out=coord_out
    )
    emission = Quantity(
        value=np.zeros((model.ncomps, len(pixel_chunk))),
        unit=u.MJy / u.sr,
    )
    for idx, component in enumerate(model.components.items()):
        integrated_comp_emission = trapezoidal(
            model=model,
            component=component,
            freq=freq,
            radial_distances=line_of_sights[component[0]],
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            color_table=color_table,
            unit_vectors=unit_vectors,
        )

        emission[idx] = integrated_comp_emission[indicies]

    return emission


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
        unit_vectors = np.asarray(
            hp.rotator.Rotator(coord=[coord_out, "E"])(unit_vectors)
        )
    return unit_vectors
