import multiprocessing
import time

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.time import Time

import zodipy

NSIDE = 256
pixels = np.arange(hp.nside2npix(NSIDE))
obs_time = Time("2020-01-01")

model = zodipy.Zodipy()
model_parallel = zodipy.Zodipy(parallel=True)

if __name__ == "__main__":
    start = time.perf_counter()
    emission = model.get_binned_emission_pix(
        40 * u.micron,
        pixels=pixels,
        nside=NSIDE,
        obs_time=obs_time,
    )
    print(
        "Time spent on a single CPU:", round(time.perf_counter() - start, 2), "seconds"
    )
    # > Time spent on a single CPU: 143.49 seconds

    start = time.perf_counter()
    emission_parallel = model_parallel.get_binned_emission_pix(
        40 * u.micron,
        pixels=pixels,
        nside=NSIDE,
        obs_time=obs_time,
    )
    print(
        f"Time spent on {multiprocessing.cpu_count()} CPUs:",
        round(time.perf_counter() - start, 2),
        "seconds",
    )
    # > Time spent on 8 CPUs: 42.52 seconds

    assert np.allclose(emission, emission_parallel)
