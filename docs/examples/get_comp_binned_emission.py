from multiprocessing import cpu_count

import astropy.units as u
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("dirbe", n_proc=cpu_count())
nside = 256

binned_emission = model.get_binned_emission_pix(
    np.arange(hp.nside2npix(nside)),
    freq=25 * u.micron,
    nside=nside,
    obs_time=Time("2022-01-01"),
    obs_pos="earth",
    return_comps=True,
)
fig = plt.figure(figsize=(8, 6.5), constrained_layout=True)
comps = ["Cloud", "Band1", "Band2", "Band3", "Ring", "Feature"]
for idx, binned_comp_emission in enumerate(binned_emission):
    hp.mollview(
        binned_comp_emission,
        title=comps[idx],
        norm="log" if idx == 0 else None,
        cmap="afmhot",
        cbar=False,
        sub=(3, 2, idx + 1),
        fig=fig,
    )
plt.show()
