from functools import lru_cache
from typing import Tuple, Union

import astropy.units as u
import numpy as np

from zodipy._source_functions import blackbody_emission_wavelen as blackbody
from zodipy.data import DATA_DIR


BANDPASS_PATH = DATA_DIR / "dirbe_spectral_response.dat"
DIRBE_BAND_REF_WAVELENS = (1.25, 2.2, 3.5, 4.9, 12, 25, 60, 100, 140, 240)


# @lru_cache
def get_dirbe_bandpass(band: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the DIRBE bandpass for a given band."""

    data = np.loadtxt(BANDPASS_PATH, skiprows=15)

    wavelen = data[:, 0].transpose()
    weights = data[:, band].transpose()
    indicies = np.nonzero(weights)

    return wavelen[indicies], weights[indicies]


def get_color_correction(T: Union[float, np.ndarray], freq: float) -> float:
    """Returns the DIREB color correction factor for a temperature and reference frequency."""

    T = np.expand_dims(T, axis=1)

    wavelen_ref = (freq * u.GHz).to("micron", equivalencies=u.spectral()).value
    closest_value = min(DIRBE_BAND_REF_WAVELENS, key=lambda x: abs(x - wavelen_ref))
    closest_idx = DIRBE_BAND_REF_WAVELENS.index(closest_value)
    band = 1 + closest_idx
    
    wavelens, weights = get_dirbe_bandpass(band)
    wavelens_ = np.expand_dims(wavelens, axis=0)
    weights = np.expand_dims(weights, axis=0)

    blackbody_ratio = (blackbody(T, wavelens_)) / blackbody(T, wavelen_ref)
    wavelen_ratio = (wavelen_ref / wavelens_) * 1e-6

    term1 = np.trapz(blackbody_ratio * weights, wavelens)
    term2 = np.trapz(wavelen_ratio * weights, wavelens)

    return term1 / term2
