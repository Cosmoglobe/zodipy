import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("dirbe")
nside = 256

binned_emission = model.get_binned_emission_pix(
    25 * u.micron,
    pixels=np.arange(hp.nside2npix(nside)),
    nside=nside,
    obs_time=Time("2020-01-01"),
    obs="earth",
    solar_cut=60 * u.deg,
)

hp.mollview(
    binned_emission,
    title="Solar cutoff at 60 degrees",
    unit="MJy/sr",
    max=80,
    coord="E",
    cmap="afmhot",
)
plt.savefig("../img/binned_solar_cutoff.png", dpi=300)
plt.show()
