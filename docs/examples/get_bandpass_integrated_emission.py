import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

nside = 128

center_freq = 800 * u.GHz
freqs = np.linspace(750, 850, 11) * u.GHz
weights = np.array([2, 3, 5, 9, 11, 11.5, 11, 9, 5, 3, 2])

plt.plot(freqs, weights)
plt.xlabel("Frequency [GHz]")
plt.ylabel("Weights")
plt.savefig("../img/bandpass.png", dpi=300)

model = Zodipy(model="planck18")

emission_central_freq = model.get_binned_emission_pix(
    freq=center_freq,
    pixels=np.arange(hp.nside2npix(nside)),
    nside=nside,
    obs_time=Time("2022-03-10"),
    obs="SEMB-L2",
)

emission_bandpass_integrated = model.get_binned_emission_pix(
    freq=freqs,
    weights=weights,
    pixels=np.arange(hp.nside2npix(nside)),
    nside=nside,
    obs_time=Time("2022-03-10"),
    obs="SEMB-L2",
)

hp.mollview(
    emission_central_freq,
    title=f"Center frequency",
    unit="MJy/sr",
    cmap="afmhot",
    norm="log",
)
plt.savefig("../img/center_freq.png", dpi=300)

hp.mollview(
    emission_bandpass_integrated,
    title="Bandpass integrated",
    unit="MJy/sr",
    cmap="afmhot",
    norm="log",
)
plt.savefig("../img/bandpass_integrated.png", dpi=300)
plt.show()
