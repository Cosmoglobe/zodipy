import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

model = Zodipy("dirbe")

latitudes = np.linspace(-90, 90, 10000) * u.deg
longitudes = np.zeros_like(latitudes)

emission = model.get_emission_ang(
    30 * u.micron,
    theta=longitudes,
    phi=latitudes,
    lonlat=True,
    obs_time=Time("2022-06-14"),
    obs="earth",
)


plt.plot(latitudes, emission)
plt.xlabel("Latitude [deg]")
plt.ylabel("Emission [MJy/sr]")
plt.savefig("../img/timestream.png", dpi=300)
plt.show()
