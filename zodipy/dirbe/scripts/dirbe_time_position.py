from pathlib import Path
from typing import List, Sequence, Tuple, Union
from functools import lru_cache

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    HeliocentricMeanEcliptic,
    GeocentricMeanEcliptic,
)
import numpy as np

DATA_DIR = Path("/Users/metinsan/Documents/doktor/zodipy/zodipy/dirbe/position/")
DIRBE_TABLE = "/Users/metinsan/Documents/doktor/zodipy/zodipy/data/dirbe_pos.dat"


@lru_cache
def read_dirbe_positions() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    files: List[Path] = []
    for file in DATA_DIR.iterdir():
        file = Path(file)
        if file.name.startswith("dmr_anc"):
            files.append(file)

    files = sorted(files)
    times_: List[float] = []
    positions_: List[Tuple[float]] = []
    for file in files:
        data = np.loadtxt(file)
        data = data[::10]

        adt = data[:, :2]
        times_.extend(adt)
        pos = data[:, 5:8]
        positions_.extend(pos)

    times = np.asarray(times_)
    positions = np.asarray(positions_)

    i4max = 4.294967296e9
    t_adt = np.zeros_like(times[:, 0])
    ind = times[:, 0] >= 0
    t_adt[ind] = times[ind, 0] + i4max * times[ind, 1]
    ind = times[:, 0] < 0
    t_adt[ind] = i4max + times[ind, 0] + i4max * times[ind, 1]
    t_adt = t_adt * (100 * u.ns)
    t_adt_d = t_adt.to("day").value  # in MJD
    mjd_times = t_adt_d

    positions *= u.m
    positions = positions.to("au")
    positions = SkyCoord(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        representation_type="cartesian",
        frame=GeocentricMeanEcliptic,
        obstime=Time(mjd_times, format="mjd"),
    )
    positions = positions.transform_to(HeliocentricMeanEcliptic)

    earth_positions = SkyCoord(
        x=np.zeros(len(mjd_times)) * u.au,
        y=np.zeros(len(mjd_times)) * u.au,
        z=np.zeros(len(mjd_times)) * u.au,
        representation_type="cartesian",
        frame=GeocentricMeanEcliptic,
        obstime=Time(mjd_times, format="mjd"),
    )
    earth_positions = earth_positions.transform_to(HeliocentricMeanEcliptic)
    return (
        mjd_times,
        np.asarray(positions.data.xyz),
        np.asarray(earth_positions.data.xyz),
    )


def tabulate_dirbe_position():
    times, positions, earths = read_dirbe_positions()
    n_steps = int(len(times) / 800)
    DIRBE_TABLE = "/Users/metinsan/Documents/doktor/zodipy/zodipy/data/dirbe_pos.dat"
    time_ = "time [MJD]"
    x_ = "x [AU]"
    y_ = "y [AU]"
    z_ = "z [AU]"
    with open(DIRBE_TABLE, "w") as file:
        file.write(f"{time_:25.15} {x_:25.15} {y_:25.15} {z_:25.15}\n")
        for time, dirbe_pos, earth_pos in zip(
            times[::n_steps],
            positions.transpose()[::n_steps],
            earths.transpose()[::n_steps],
        ):

            file.write(
                f"{time:<25.15} {dirbe_pos[0]:<25.15} {dirbe_pos[1]:<25.15} {dirbe_pos[2]:<25.15}\n"
            )


def get_dirbe_position(time: Union[float, Sequence[float]]) -> Union[List[float], List[List[float]]]:
    times, *position = np.loadtxt(DIRBE_TABLE, dtype=float, unpack=True, skiprows=(1))

    return [np.interp(time, times, position[idx]) for idx in range(3)]


def get_earth_position(time: Union[float, Sequence[float]]) -> Union[List[float], List[List[float]]]:
    times, *_ = np.loadtxt(DIRBE_TABLE, dtype=float, unpack=True, skiprows=(1))
    earth_positions = SkyCoord(
        x=np.zeros(len(times)) * u.au,
        y=np.zeros(len(times)) * u.au,
        z=np.zeros(len(times)) * u.au,
        representation_type="cartesian",
        frame=GeocentricMeanEcliptic,
        obstime=Time(times, format="mjd"),
    )
    earth_positions = earth_positions.transform_to(HeliocentricMeanEcliptic)

    return [np.interp(time, times, np.asarray(earth_positions.data.xyz)[idx]) for idx in range(3)]


if __name__ == "__main__":
    print(get_dirbe_position([47871, 47876]))
