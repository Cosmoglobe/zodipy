from typing import List
import astropy.units as u
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from zodipy import Zodipy
from zodipy._labels import CompLabel
from zodipy._dirbe import DIRBE_BAND_REF_WAVELENS
from zodipy.dirbe.scripts.dirbe_time_position import get_earth_position



# This is my path to the TODS, you need to specify this your self.
PATH_TO_TODS = "/Users/metinsan/Documents/doktor/data/dirbe/tod/"

BAND = 6
nside = 128
npix = hp.nside2npix(nside)
freq = DIRBE_BAND_REF_WAVELENS[BAND - 1] * u.micron

model = Zodipy()


# Set emisivities to 1.0
for label in model._model.spectral_params["emissivities"]:
    if isinstance(label, CompLabel):
        model._model.spectral_params["emissivities"][label] = tuple([
            1.0 for _ in range(len(DIRBE_BAND_REF_WAVELENS))
        ])

def save_timestreams(band: int):
    """Saves the DIRBE and corresponding zodipy simulated timestream to files."""

    DATA_PATH = f"{PATH_TO_TODS}/Phot{band:02}.hdf5"

    timestreams: List[NDArray[np.float32]] = []
    zodipy_timestreams: List[NDArray[np.float32]] = []

    with h5py.File(DATA_PATH, "r") as file:

        for tod_chunk in tqdm(file):
            tods = np.asarray(file[f"{tod_chunk}/A/tod"][()], dtype=np.float32)
            pixels = np.asarray(file[f"{tod_chunk}/A/pix"][()], dtype=np.int32)
            times = np.asarray(file[f"{tod_chunk}/A/time"][()], dtype=np.float32)

            condition = tods > 0
            filtered_tods = tods[condition]

            # Some TOD chuncks have only flagged data.
            if not filtered_tods.size > 0:
                continue

            filtered_pixels = pixels[condition]
            filtered_times = times[condition]
            time = filtered_times[int(len(filtered_times)/2)]
            
            earth_pos = get_earth_position(time) * u.AU

            zodipy_tods = model.get_emission(
                freq,
                nside=nside,
                pixels=filtered_pixels,
                obs_pos=earth_pos,
                return_comps=True,
                dirbe_colorcorr=True,
            )
   
            timestreams.append(filtered_tods)
            zodipy_timestreams.append(zodipy_tods)

        full_timestream = np.concatenate(timestreams)
        full_zodipy_timestream = np.concatenate(zodipy_timestreams, axis=1)

        np.save(f"DIRBE_timestream_band{BAND}.npy", full_timestream)
        np.save(f"zodipy_timestream_band{BAND}_unit_emissivity.npy", full_zodipy_timestream.value)

save_timestreams(band=BAND)

    