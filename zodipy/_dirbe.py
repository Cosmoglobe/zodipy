from typing import Sequence, Tuple, Union

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy.data import DATA_DIR
from zodipy._source_functions import blackbody_emission_lambda


BANDPASS_PATH = DATA_DIR / "dirbe_spectral_response.dat"
DIRBE_BAND_REF_WAVELENS = (1.25, 2.2, 3.5, 4.9, 12.0, 25.0, 60.0, 100.0, 140.0, 240.0)
EFFECTIVE_BANDPASS_REF_FREQS = (
    59.5,
    22.4,
    22.0,
    8.19,
    13.3,
    4.13,
    2.32,
    0.974,
    0.605,
    0.495,
)


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

    blackbody_ratio = (
        blackbody_emission_lambda(T, wavelens_)
    ) / blackbody_emission_lambda(T, wavelen_ref)
    wavelen_ratio = (wavelen_ref / wavelens_) * 1e-6

    term1 = np.trapz(blackbody_ratio * weights, wavelens)
    term2 = np.trapz(wavelen_ratio * weights, wavelens)

    return term1 / term2


def tabulate_color_correction() -> None:
    T = np.expand_dims(np.linspace(50, 1000, 200), axis=1)
    color_corrs = np.zeros((10, len(T)))
    for band in range(1, 10):
        wavelens, weights = get_dirbe_bandpass(band)
        wavelens_ = np.expand_dims(wavelens, axis=0)
        weights = np.expand_dims(weights, axis=0)

        wavelen_ref = DIRBE_BAND_REF_WAVELENS[band - 1]
        blackbody_ratio = (
            blackbody_emission_lambda(T, wavelens_)
        ) / blackbody_emission_lambda(T, wavelen_ref)
        wavelen_ratio = (wavelen_ref / wavelens_) * 1e-6

        term1 = np.trapz(blackbody_ratio * weights, wavelens)
        term2 = np.trapz(wavelen_ratio * weights, wavelens)
        color_corrs[band - 1] = term1 / term2

    with open(f"{DATA_DIR}/dirbe_color_corr.dat", "w") as file:
        file.write(
            f"T\t\t band 1\t\t band 2\t\t band 3\t\t band 4\t\t band 5\t\t band 6\t\t band 7\t\t band 8\t\t band 9\t\t band 10\n"
        )
        temps = T.squeeze()
        for idx in range(len(temps)):
            str_ = f"{temps[idx]:<10.2f} {color_corrs[0, idx]:<10.5f} {color_corrs[1, idx]:<10.5f} {color_corrs[2, idx]:<10.5f} {color_corrs[3, idx]:<10.5f} {color_corrs[4, idx]:<10.5f} {color_corrs[5, idx]:<10.5f} {color_corrs[6, idx]:<10.5f} {color_corrs[7, idx]:<10.5f} {color_corrs[8, idx]:<10.5f} {color_corrs[9, idx]:<10.5f}"
            str_ += "\n"
            file.write(str_)


def read_color_corr(band: int) -> NDArray[np.float64]:
    """Reads in DIRBE color correction factors."""

    return np.loadtxt(
        f"{DATA_DIR}/dirbe_color_corr.dat",
        skiprows=1,
        usecols=(0, band),
    )


def get_normalized_weights(
    freqs: Sequence[float], weights: Sequence[float]
) -> Sequence[float]:
    """Returns a normalized bandpass weights."""

    return weights / np.trapz(weights, freqs)
