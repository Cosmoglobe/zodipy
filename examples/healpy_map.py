import multiprocessing

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

model = zodipy.Model(30 * u.micron)

nside = 256
pixels = np.arange(hp.nside2npix(nside))
lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

skycoord = SkyCoord(lon, lat, unit=u.deg, frame="galactic", obstime=Time("2022-01-14"))

emission = model.evaluate(skycoord, nprocesses=multiprocessing.cpu_count())

hp.mollview(
    emission,
    unit="MJy/sr",
    cmap="afmhot",
    min=0,
    max=80,
    title="Zodiacal light at 30 µm (2022-01-14)",
)
plt.savefig("../img/healpix_map.png", dpi=300, bbox_inches="tight")
plt.show()
