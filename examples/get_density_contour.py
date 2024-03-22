import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from zodipy import tabulate_density

N = 200

x = np.linspace(-5, 5, N)  # x-plane
y = np.linspace(-5, 5, N)  # y-plane
z = np.linspace(-2, 2, N)  # z-plane

grid = np.asarray(np.meshgrid(x, y, z))
density_grid = tabulate_density(grid, model="dirbe")
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
plt.title("Cross section of the interplanetary dust density (yz-plane)")
plt.xlabel("x [AU]")
plt.ylabel("z [AU]")
# plt.savefig("../img/density_grid.png", dpi=300)
plt.show()
