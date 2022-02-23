from typing import List
import astropy.units as u
import emcee 
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from zodipy import Zodipy
from zodipy._labels import CompLabel
from zodipy.fitting.fit_template import fit_template
from zodipy.fitting.evaluate_likelihood import eval_like
from zodipy._dirbe import DIRBE_BAND_REF_WAVELENS
from zodipy.dirbe.scripts.dirbe_time_position import get_earth_position

np.random.seed(420)

# This is my path to the TODS, you need to specify this your self.
PATH_TO_TODS = "/home/daniel/data/dirbe/h5/"

# Can find the OG maps on OLA. (also smoothed and ud_graded @ /mn/stornext/d14/Planck1/daniher/data/zodi/zodipy_data/
template_file= "/home/daniel/data/npipe6v20/npipe6v20_857_map_n0128_42arcmin_uK.fits"

template     = hp.read_map(template_file)
template     = template-np.min(template)

ecl_template = np.asarray(hp.rotator.Rotator(coord=["G","E"]).rotate_map_pixel(template))

model = Zodipy()

BAND = 6
nside = 128
npix = hp.nside2npix(nside)
freq = DIRBE_BAND_REF_WAVELENS[BAND - 1] * u.micron

def fit_emissivities(band: int):
    """Currently this only binnes the TODS."""

    DATA_PATH = f"{PATH_TO_TODS}/Phot{band:02}.hdf5"


    timestreams: List[NDArray[np.float32]] = []
    zodipy_timestreams: List[NDArray[np.float32]] = []

    with h5py.File(DATA_PATH, "r") as file:

        for idx, tod_chunk in enumerate(tqdm(file)):
            tods = np.asarray(file[f"{tod_chunk}/A/tod"][()], dtype=np.float32)
            pixels = np.asarray(file[f"{tod_chunk}/A/pix"][()], dtype=np.int32)
            times = np.asarray(file[f"{tod_chunk}/A/time"][()], dtype=np.float32)

            condition = tods > 0
            filtered_tods = tods[condition]
            filtered_pixels = pixels[condition]
            filtered_times = times[condition]
            time = filtered_times[int(len(filtered_times)/2)]
            earth_pos = get_earth_position(time) * u.AU

            zodipy_tods = model.get_time_ordered_emission(
                freq,
                nside=nside,
                pixels=filtered_pixels,
                observer_pos=earth_pos,
                return_comps=True,
                color_corr=True,
            )
   
            timestreams.append(filtered_tods)
            zodipy_timestreams.append(zodipy_tods)

            plt.plot(timestreams[idx], label="dirbe")
            plt.plot(zodipy_timestreams[idx].sum(axis=0), label="zodipy")
            plt.show()


# BIG DANIEL TESTING BLOCK
# import numba

# @numba.njit
# def accumuate_tods(emission, pixels, tods):
#     for i in range(len(tods)):
#         emission[pixels[i]] += tods[i]

#     return emission
# def fit_emissivities(band: int):
#     """Currently this only binnes the TODS."""

#     DATA_PATH = f"{PATH_TO_TODS}/Phot{band:02}.hdf5"

#     with h5py.File(DATA_PATH, "r") as file:
#         npix = hp.nside2npix(nside)

#         emission = np.zeros(npix)
#         hits = np.zeros_like(emission)

#         for tod_chunk in tqdm(file):
#             pixels = np.asarray(file[f"{tod_chunk}/A/pix"][()])
#             tod = np.asarray(file[f"{tod_chunk}/A/tod"][()], dtype=np.float64)

#             condition = tod > 0
#             filtered_tods = tod[condition]
#             filtered_pixels = pixels[condition]
#             emission = accumuate_tods(
#                 emission=emission,
#                 pixels=filtered_pixels,
#                 tods=filtered_tods,
#             )
#             unique_pixels, pixel_counts = np.unique(
#                 filtered_pixels, return_counts=True
#             )

#             hits[unique_pixels] += pixel_counts

#     return emission, hits


# emission, hits = fit_emissivities(band=BAND)
# emission /= hits
# mask     = (hits > 0.0)

# ngibbs = 1000

# # Check the likelihood for a range of amplitudes
# amps = np.zeros(ngibbs)
# lnL  = np.zeros(ngibbs)
# x    = np.linspace(1,ngibbs,ngibbs)

# for i in tqdm(range(ngibbs)):

#     amps[i] = fit_template(emission,hits,ecl_template,mask,sample=True)
#     fitmap = amps[i]*ecl_template
#     lnL[i] = eval_like(emission,hits,fitmap,mask)

# fig, ax = plt.subplots(2,1)

# ax[0].plot(x,amps)
# ax[1].plot(x,lnL)
# plt.show()

# wurr = np.where(lnL == np.max([lnL]))

# print(amps[wurr])

# res = emission-amps[wurr]*ecl_template

# mean = np.mean(res[mask])
# std  = np.std(res[mask])


# hp.mollview(res,min=mean-std,max=mean+std)
# plt.show()

# exit()
    
# a_mean = np.mean(amps)
# a_std  = np.std(amps)

# print(f"amplitude mean: {a_mean}, amplitude std: {a_std}")

# dust_corr_map = emission-a_mean*ecl_template

# mean    = np.mean(dust_corr_map[mask]) 
# std     = np.std(dust_corr_map[mask])
# print(mean,std)
