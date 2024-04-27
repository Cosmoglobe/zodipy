from multiprocessing import cpu_count

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("planck18", n_proc=cpu_count())
nside = 256

binned_emission = model.get_binned_emission_pix(
    np.arange(hp.nside2npix(nside)),
    freq=857 * u.GHz,
    nside=nside,
    obs_time=Time("2022-06-14"),
    obs_pos="earth",
)

hp.mollview(
    binned_emission,
    title="Binned zodiacal emission at 857 GHz",
    unit="MJy/sr",
    min=0,
    max=1,
    cmap="afmhot",
)
plt.show()
