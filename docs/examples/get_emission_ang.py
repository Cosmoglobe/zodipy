from multiprocessing import cpu_count

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("dirbe", n_proc=cpu_count())

latitudes = np.linspace(-90, 90, 10000) * u.deg
longitudes = np.zeros_like(latitudes)

emission = model.get_emission_ang(
    theta=longitudes,
    phi=latitudes,
    freq=30 * u.micron,
    lonlat=True,
    obs_time=Time("2022-06-14"),
    obs_pos="earth",
)


plt.plot(latitudes, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.show()
