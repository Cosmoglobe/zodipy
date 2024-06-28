import multiprocessing

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

COMP_NAMES = [
    "Smooth cloud",
    "Dust band 1",
    "Dust band 2",
    "Dust band 3",
    "Circum-solar Ring",
    "Earth-trailing Feature",
]

model = zodipy.Model(24 * u.micron)

nside = 32
pixels = np.arange(hp.nside2npix(nside))
lon, lat = hp.pix2ang(nside, pixels, lonlat=True)

skycoord = SkyCoord(
    lon,
    lat,
    unit=u.deg,
    frame="barycentricmeanecliptic",
    obstime=Time("2022-01-14"),
)

emission = model.evaluate(skycoord, return_comps=True, nprocesses=multiprocessing.cpu_count())

fig = plt.figure(figsize=(8, 7))
for idx, comp_emission in enumerate(emission):
    hp.mollview(
        comp_emission,
        title=COMP_NAMES[idx],
        norm="log" if idx == 0 else None,
        cmap="afmhot",
        cbar=False,
        sub=(3, 2, idx + 1),
        fig=fig,
    )
plt.savefig("../img/component_maps.png", dpi=250, bbox_inches="tight")
plt.show()
