import multiprocessing
import time

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.time import Time

import zodipy

nside = 256
pixels = np.arange(hp.nside2npix(nside))
obs_time = Time("2020-01-01")

model = zodipy.Zodipy(parallel=False)
model_parallel = zodipy.Zodipy()

start = time.perf_counter()
emission = model.get_binned_emission_pix(
    40 * u.micron,
    pixels=pixels,
    nside=nside,
    obs_time=obs_time,
)
print("Time spent on a single CPU:", round(time.perf_counter() - start, 2), "seconds")
# > Time spent on a single CPU: 91.76 seconds

start = time.perf_counter()
emission_parallel = model_parallel.get_binned_emission_pix(
    40 * u.micron,
    pixels=pixels,
    nside=nside,
    obs_time=obs_time,
)
print(
    f"Time spent on {multiprocessing.cpu_count()} CPUs:",
    round(time.perf_counter() - start, 2),
    "seconds",
)
# > Time spent on 8 CPUs: 26.87 seconds

assert np.allclose(emission, emission_parallel)
