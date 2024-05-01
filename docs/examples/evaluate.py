import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from zodipy import Model

model = Model()

# Longitude and Latitude values corresponding to a scan through the eclitpic plane
lats = np.linspace(-90, 90, 100) * u.deg
lons = np.zeros_like(lats)

obs_time = Time("2022-06-14")

# The SkyCoord object needs to include the coordinate frame and time of observation
coords = SkyCoord(
    lons,
    lats,
    frame=BarycentricMeanEcliptic,
    obstime=obs_time,
)

emission = model.evaluate(coords, freq=30 * u.micron)

plt.plot(lats, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.show()
