import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from matplotlib.colors import LogNorm

from zodipy import grid_number_density

N = 200

x = np.linspace(-5, 5, N) * u.AU  # x-plane
y = np.linspace(-5, 5, N) * u.AU  # y-plane
z = np.linspace(-2, 2, N) * u.AU  # z-plane

density_grid = grid_number_density(
    x,
    y,
    z,
    obstime=Time("2021-01-01T00:00:00", scale="utc"),
    name="DIRBE",
)
density_grid = density_grid.sum(axis=0)  # Sum over all components

plt.pcolormesh(
    x,
    y,
    density_grid[N // 2].T,  # cross section in the yz-plane
    cmap="afmhot",
    norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
    shading="gouraud",
    rasterized=True,
)
plt.title("Cross section of the number density in the DIRBE model")
plt.xlabel("x [AU]")
plt.ylabel("z [AU]")
plt.savefig("../img/number_density.png", dpi=300, bbox_inches="tight")
plt.show()
