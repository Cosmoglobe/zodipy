import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time, TimeDelta

from zodipy import Model

model = Model(30 * u.micron)

lats = np.linspace(-90, 90, 100) * u.deg
lons = np.zeros_like(lats)

t0 = Time("2022-06-14")
dt = TimeDelta(1, format="sec")
obstimes = t0 + np.arange(lats.size) * dt

coords = SkyCoord(
    lons,
    lats,
    frame=BarycentricMeanEcliptic,
    obstime=obstimes,
)

emission = model.evaluate(coords)

plt.plot(lats, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.savefig("../img/ecliptic_scan.png", dpi=300, bbox_inches="tight")
plt.show()
