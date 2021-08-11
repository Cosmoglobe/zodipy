import datetime
import functools
from math import sin, cos
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
def get_target_coordinates(target: str, time: datetime.datetime.date) -> np.ndarray:
    """Returns the heliocentric cartesian coordinates of the target.
    
    Parameters
    ----------
    target : str
        Name of the target body.
    time : `datetime.datetime`
        Date and time at which to get the targets coordinates.

    Returns
    -------
    `numpy.ndarray`
        Heliocentric cartesian coordinates of the target.
    """

    if target.lower() in TARGET_ALIASES:
        target = TARGET_ALIASES[target.lower()]
    else:
        warnings.warn(
            'The Zodiacal Emission model is only valid in the immediate '
            'surroundings of the earth'
        )

    stop = time + datetime.timedelta(days=1)
    start = time
    epochs = dict(start=str(start), stop=str(stop), step='1d')
    
    query = Horizons(id=target, id_type='majorbody', location='c@sun', epochs=epochs)
    ephemerides = query.ephemerides()

    R = ephemerides['r'][0]
    lon, lat = ephemerides['EclLon'][0], ephemerides['EclLat'][0]

    x = R * cos(lat) * cos(lon)
    y = R * cos(lat) * sin(lon)
    z = R * sin(lat)
    
    return np.array([[x], [y], [z]])


def change_coordinate_system(
    input_map: np.ndarray, coord_out: str, coord_in: str = 'E'
) -> np.ndarray:
    """Rotates a map from coordinate system to another using healpy.Rotator
    
    Paramters
    ---------
    input_map : `numpy.ndarray`
        Map to rotate.
    coord_out : str
        Coordinate system of the output_map.
    coord_in : str
        Coordinate system of the input_map.
    Returns
    -------
    output_map : `numpy.ndarray`
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