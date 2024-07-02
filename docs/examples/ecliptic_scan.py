import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import BarycentricMeanEcliptic, SkyCoord
from astropy.time import Time

from zodipy import Model

model = Model(30 * u.micron)

# Longitude and Latitude values corresponding to a scan through the eclitpic plane
lats = np.linspace(-90, 90, 100) * u.deg
lons = np.zeros_like(lats)

coords = SkyCoord(
    lons,
    lats,
    frame=BarycentricMeanEcliptic,
    obstime=Time("2022-06-14"),
)

emission = model.evaluate(coords)

plt.plot(lats, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.savefig("../img/ecliptic_scan.png", dpi=300, bbox_inches="tight")
plt.show()
