import multiprocessing

import astropy.units as u
import astropy_healpix as ahp
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

import zodipy

model = zodipy.Model(30 * u.micron, n_proc=multiprocessing.cpu_count())

healpix = ahp.HEALPix(nside=256, frame="galactic")
pixels = np.arange(healpix.npix)
skycoord = healpix.healpix_to_skycoord(pixels)

# Note that we manually set the obstime attribute
skycoord.obstime = Time("2022-01-14")

emission = model.evaluate(skycoord)

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
