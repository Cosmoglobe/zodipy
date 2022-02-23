import astropy.units as u
import emcee 
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numba
import tqdm

from zodipy import Zodipy
from zodipy._labels import CompLabel

model = Zodipy()

BAND = 6
nside = 128


@numba.njit
def accumuate_tods(emission, pixels, tods):
    for i in range(len(tods)):
        emission[pixels[i]] += tods[i]

    return emission

def fit_emissivities(band: int):
    DATA_PATH = f"/Users/metinsan/Documents/doktor/data/dirbe/tod/Phot{band:02}.hdf5"

    with h5py.File(DATA_PATH, "r") as file:
        npix = hp.nside2npix(nside)

        emission = np.zeros(npix)
        hits = np.zeros_like(emission)

        for tod_chunk in tqdm(file):
            pixels = np.asarray(file[f"{tod_chunk}/A/pix"][()])
            tod = np.asarray(file[f"{tod_chunk}/A/tod"][()], dtype=np.float64)

            condition = tod > 0
            filtered_tods = tod[condition]
            filtered_pixels = pixels[condition]
            emission = accumuate_tods(
                emission=emission,
                pixels=filtered_pixels,
                tods=filtered_tods,
            )
            unique_pixels, pixel_counts = np.unique(
                filtered_pixels, return_counts=True
            )

            hits[unique_pixels] += pixel_counts

    return emission, hits



emission, hits = fit_emissivities()(band=BAND)
hp.mollview(hits, norm="hist")
plt.show()
emission /= hits

hp.mollview(emission, min=0.001,max=80)
plt.show()
    