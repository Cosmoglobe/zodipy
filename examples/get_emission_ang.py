import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from zodipy import Zodipy

model = Zodipy("dirbe")

theta = np.linspace(0, np.pi, 10000) * u.rad
phi = np.zeros_like(theta)

emission = model.get_emission_ang(
    30 * u.micron,
    theta=theta,
    phi=phi,
    obs_time=Time("2022-06-14"),
    obs="earth",
)


plt.plot(emission)
plt.xlabel("Theta [rad]")
plt.ylabel("Emission [MJy/sr]")
plt.savefig("../img/timestream.png", dpi=300)
plt.show()
