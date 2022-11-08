import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from zodipy import tabulate_density

N = 300

x = np.linspace(-6, 6, N)  # x-plane
y = np.linspace(-6, 6, N)  # y-plane
z = np.linspace(-2.5, 2.5, N)  # z-plane

grid = np.asarray(np.meshgrid(x, y, z, indexing="ij"))
density_grid = tabulate_density(grid, model="rrm")
density_grid = density_grid.sum(axis=0)  # Sum over all components
dens = density_grid[N // 2].T
plt.pcolormesh(
    x,
    z,
    dens,  # cross section in the yz-plane
    cmap="afmhot",
    # vmin=-2,
    # vmax=0.5,
    norm=LogNorm(vmin=density_grid.min(), vmax=density_grid.max()),
    shading="gouraud",
    rasterized=True,
)
# plt.contour(
#     x, z, dens, levels=[-1.5, -1, -0.5, 0, 0.5], colors="k", linewidths=0.5, linestyles="solid"
# )
plt.title("RRM all comps")
# plt.title("Cross section of the interplanetary dust density (yz-plane)")
plt.xlabel("x [AU]")
plt.ylabel("z [AU]")
# plt.savefig("../img/density_grid.png", dpi=300)
plt.show()
