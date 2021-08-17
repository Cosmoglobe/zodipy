from datetime import datetime, timedelta
import functools
import warnings

from astroquery.jplhorizons import Horizons
import healpy as hp
import numpy as np


TARGET_ALIASES = {
    'l1' : 'SEMB-L1',
    'l2' : 'SEMB-L2',
    'planck' : 'Planck',
    'wmap' : 'WMAP',
    'earth' : 'Earth-Moon Barycenter',
}


@functools.lru_cache
def get_target_coordinates(
    target: str, start: datetime, stop:datetime = None, step: str = '1d'
) -> np.ndarray:
    """Returns the heliocentric cartesian coordinates of the target.
    
    Parameters
    ----------
    target
        Name of the target body.
    time
        Date and time at which to get the targets coordinates.

    Returns
    -------
        Heliocentric cartesian coordinates of the target.
    """

    if target.lower() in TARGET_ALIASES:
        target = TARGET_ALIASES[target.lower()]
    else:
        warnings.warn(
            'The K98 model is only valid in the immediate surroundings of'
            'the earth'
        )

    if stop is None:
        stop_ = start + timedelta(1)
    else:
        stop_ = stop

    epochs = dict(start=str(start), stop=str(stop_), step=f'{step}')
    query = Horizons(id=target, id_type='majorbody', location='c@sun', epochs=epochs)
    ephemerides = query.ephemerides()

    R = ephemerides['r'].value
    lon, lat = ephemerides['EclLon'].value, ephemerides['EclLat'].value
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    coordinates = np.stack((x,y,z), axis=1)

    if stop is None:
        return np.expand_dims(coordinates[0], axis=0)
    return coordinates


def change_coordinate_system(
    input_map: np.ndarray, coord_out: str, coord_in: str = 'E'
) -> np.ndarray:
    """Rotates a map from coordinate system to another using healpy.Rotator
    
    Parameters
    ----------
    input_map
        Map to rotate.
    coord_out
        Coordinate system of the output_map.
    coord_in
        Coordinate system of the input_map.

    Returns
    -------
    output_map
        Rotated map.
    """

    if coord_in == coord_out:
        return input_map
    
    nside = hp.get_nside(input_map)
    npix = hp.nside2npix(nside)
    
    rotator = hp.Rotator(coord=[coord_out, coord_in])
    new_angles = rotator(hp.pix2ang(nside, np.arange(npix)))
    new_pixels = hp.ang2pix(nside, *new_angles)
    output_map = input_map[..., new_pixels]

    return output_map