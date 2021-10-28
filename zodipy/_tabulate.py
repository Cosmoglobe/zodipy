import datetime
from typing import Callable, List, Optional, Sequence
import h5py

from astropy.time import Time
import healpy as hp
import numpy as np
from scipy.interpolate import interp1d

import zodipy


DATA_DIR = "/Users/metinsan/Documents/doktor/zodipy/zodipy/data/"
TABLE = DATA_DIR + "zodi_table64.h5"

INITIAL_DAY = datetime.datetime(2020, 1, 1)
DAYS_IN_A_YEAR = 365
DAYS_STEP = 40

def get_JD_range(n: int = DAYS_STEP):
    """Returns a range of astropy dates in JD.
    
    Parameters
    ----------
    n
        number of Julian dates to return.

    Returns
    -------
        List of julian days that cover a year with stepsize n.
    """

    n_days = round(DAYS_IN_A_YEAR / n)

    dates = [
        (INITIAL_DAY + datetime.timedelta(days=day)).isoformat()
        for day in range(1, DAYS_IN_A_YEAR, n_days)
    ]

    return Time(dates, format="isot").jd


def JD_to_yday(date: float):
    """Converts a Julian date to yday.
    
    Parameters
    ----------
    date
        Julian date.

    Returns
    -------
    yday
        Julian date converted to the day number of that year.
    """

    date = Time(date, format="jd").to_datetime()
    return date.timetuple().tm_yday


def tabulate(
    nside: int,
    freqs: Sequence[float],
    dates: Sequence[str],
    model: str,
    observer: str = "L2",
    filename: str = TABLE,
) -> None:
    """Function that tabulates simulations to file.
    
    The simulations are created using the PixelWeightedMeanStrategy.
    
    Parameters
    ----------
    nside
        Healpix resolution parameter.
    freqs
        Frequencies for which to compute and tabulate the Zodi emission.
    dates
        Julian dates for each simulated observation that is tabulated.
    observer
        String representing an observer in the solar system.
    filename
        Name of the tabulated file.
    """

    with h5py.File(filename, "a") as file:
        model_group = file.create_group(model)
        for freq in freqs:
            print(freq)
            freq_group = model_group.create_group(f"{int(freq):04}")
            for i, date in enumerate(dates):
                zodi = zodipy.Zodi(observer=observer, epochs=date, model=model)
                
                emission = zodi.get_emission(nside=nside, freq=freq, return_comps=True, coord="E")
                data_set = freq_group.create_dataset(f"{i:04d}", data=emission)
                data_set.attrs["day"] = JD_to_yday(date)


def get_tabulated_data(
    nside: int,
    freq: float,
    model: str,
) -> Callable[[int], np.ndarray]:
    """Creates and returns a 1D interpolater from tabulated zodi data."""

    frequencies = []
    days = []
    n_comps = 0
    with h5py.File(TABLE, "r") as file:
        if model not in file:
            raise ModuleNotFoundError(f"tabulated data not found for model {model!r}")
        data = file[model]
        for freq in data:
            frequencies.append(float(freq))
            if not days:
                for day in data[freq]:
                    if n_comps == 0:
                        n_comps = data[freq][day][()].shape[0]
                    days.append(data[freq][day].attrs["day"])

        frequencies = np.asarray(frequencies)
        days = np.asarray(days)
        simulations = np.zeros((len(frequencies), len(days), n_comps, hp.nside2npix(nside)))
        for i, freq in enumerate(data):
            for j, day in enumerate(data[freq]):
                simulations[i, j] = data[freq][day][()]

    return frequencies, days, simulations


if __name__ == "__main__":
    tabulate(
        nside=64,
        freqs=np.geomspace(100,5000, 50),
        dates=get_JD_range(),
        model="K98",
        filename=DATA_DIR + "zodi_table64.h5"
    )
