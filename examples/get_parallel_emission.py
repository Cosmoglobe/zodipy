import time

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.time import Time

from zodipy import Zodipy

nside = 256
pixels = np.arange(hp.nside2npix(nside))
obs_time = Time("2020-01-01")
n_proc = 8

model = Zodipy()
model_parallel = Zodipy(n_proc=n_proc)

start = time.perf_counter()
emission = model.get_binned_emission_pix(
    40 * u.micron,
    pixels=pixels,
    nside=nside,
    obs_time=obs_time,
)
print("Time spent on a single CPU:", round(time.perf_counter() - start, 2), "seconds")
# > Time spent on a single CPU: 35.23 seconds

start = time.perf_counter()
emission_parallel = model_parallel.get_binned_emission_pix(
    40 * u.micron,
    pixels=pixels,
    nside=nside,
    obs_time=obs_time,
)
print(
    f"Time spent on {n_proc} CPUs:",
    round(time.perf_counter() - start, 2),
    "seconds",
)
# > Time spent on 8 CPUs: 12.85 seconds

assert np.allclose(emission, emission_parallel)
