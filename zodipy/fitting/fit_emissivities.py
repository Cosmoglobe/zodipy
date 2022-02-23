import astropy.units as u
import emcee 
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import numba
from tqdm import tqdm

from zodipy import Zodipy
from zodipy._labels import CompLabel
from zodipy.fitting.fit_template import fit_template

model = Zodipy()

BAND = 10
nside = 128

# This is my path to the TODS, you need to specify this your self.
PATH_TO_TODS = "/home/daniel/data/dirbe/h5/"

# Can find the OG maps on OLA. (also smoothed and ud_graded @ /mn/stornext/d14/Planck1/daniher/data/zodi/zodipy_data/
template_file= "/home/daniel/data/npipe6v20/npipe6v20_857_map_n0128_42arcmin_uK.fits"

template     = hp.read_map(template_file)
template     = template-np.min(template)

ecl_template = np.asarray(hp.rotator.Rotator(coord=["G","E"]).rotate_map_pixel(template))

@numba.njit
def accumuate_tods(emission, pixels, tods):
    for i in range(len(tods)):
        emission[pixels[i]] += tods[i]

    return emission
    
def fit_emissivities(band: int):
    """Currently this only binnes the TODS."""

    DATA_PATH = f"{PATH_TO_TODS}/Phot{band:02}.hdf5"

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



emission, hits = fit_emissivities(band=BAND)
emission /= hits
mask     = (hits > 0.0)

ngibbs = 1000

amps = np.zeros(ngibbs)

for i in tqdm(range(ngibbs)):

    amps[i] = fit_template(emission,hits,ecl_template,mask,sample=True)

a_mean = np.mean(amps)
a_std  = np.std(amps)

print(f"amplitude mean: {a_mean}, amplitude std: {a_std}")

dust_corr_map = emission-a_mean*ecl_template

mean    = np.mean(dust_corr_map[mask]) 
std     = np.std(dust_corr_map[mask])
print(mean,std)

hp.mollview(emission,min=(mean-std),max=(mean+std))
plt.show()

hp.mollview(dust_corr_map,min=(mean-std),max=(mean+std))
plt.show()
    
