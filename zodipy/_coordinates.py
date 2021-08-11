import datetime

from astroquery.jplhorizons import Horizons
import numpy as np

TARGET_ALIASES = {
    'l1' : 'SEMB-L1',
    'l2' : 'SEMB-L2',
    'planck' : 'Planck',
    'wmap' : 'WMAP',
    'earth' : 'Earth-Moon Barycenter',
}


def get_target_coordinates(target: str, time: datetime.datetime) -> np.ndarray:
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

    stop = time + datetime.timedelta(days=1)
    start = time
    epochs = dict(
        start=str(start), stop=str(stop), step='1d'
    )
    query = Horizons(id=target, id_type='majorbody', location='c@sun', epochs=epochs)
    ephemerides = query.ephemerides()

    R = ephemerides['r'][0]
    lon = ephemerides['EclLon'][0]
    lat = ephemerides['EclLat'][0]

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    return np.array([[x], [y], [z]])