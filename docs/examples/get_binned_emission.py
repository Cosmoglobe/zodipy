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
)

hp.mollview(
    binned_emission,
    title="Binned zodiacal emission at 857 GHz",
    unit="MJy/sr",
    min=0,
    max=1,
    cmap="afmhot",
)
plt.savefig("../img/binned.png", dpi=300)
plt.show()
