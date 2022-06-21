import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("planck18")
nside = 256

binned_emission = model.get_binned_emission_pix(
    857 * u.GHz,
    pixels=np.arange(hp.nside2npix(nside)),
    nside=nside,
    obs_time=Time("2022-06-14"),
    obs="earth",
    coord_in="G",   # Coordinates of the input pointing
)

hp.mollview(
    binned_emission,
    title="Binned Zodiacal emission  at 857 GHz",
    unit="MJy/sr",
    coord="G",
    max=1,
    norm="log",
    cmap="afmhot"
)
hp.graticule(coord="E")
plt.savefig("../img/binned_gal.png", dpi=300)
plt.show()
