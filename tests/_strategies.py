from __future__ import annotations

import datetime
from functools import partial
from math import log2
from typing import Any, Callable, Literal, Sequence

import astropy.coordinates as coords
import astropy.units as u
import astropy_healpix as hp
import numpy as np
import numpy.typing as npt
from astropy import time
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    booleans,
    composite,
    datetimes,
    floats,
    integers,
    lists,
    one_of,
    sampled_from,
)

import zodipy
from zodipy._line_of_sight import COMPONENT_CUTOFFS
from zodipy.model_registry import model_registry

MIN_FREQ = u.Quantity(10, u.GHz)
MAX_FREQ = u.Quantity(0.1, u.micron).to(u.GHz, equivalencies=u.spectral())
N_FREQS = 1000
FREQ_LOG_RANGE = np.geomspace(np.log(MIN_FREQ.value), np.log(MAX_FREQ.value), N_FREQS).tolist()

MIN_DATE = datetime.datetime(year=1900, month=1, day=1)
MAX_DATE = datetime.datetime(year=2100, month=1, day=1)

MIN_NSIDE = 8
MAX_NSIDE = 1024
MIN_NSIDE_EXP = int(log2(MIN_NSIDE))
MAX_NSIDE_EXP = int(log2(MAX_NSIDE))

MAX_PIXELS_LEN = 10000
MAX_ANGELS_LEN = 10000

AVAILABLE_MODELS = zodipy.model_registry.models


@composite
def quantities(
    draw: Callable[[SearchStrategy[float]], u.Quantity],
    min_value: float,
    max_value: float,
    unit: u.Unit,
) -> u.Quantity:
    return draw(
        floats(
            min_value=min_value,
            max_value=max_value,
            allow_infinity=False,
            allow_nan=False,
        ).map(partial(u.Quantity, unit=unit))
    )


@composite
def times(draw: DrawFn) -> time.Time:
    return draw(datetimes(min_value=MIN_DATE, max_value=MAX_DATE).map(time.Time))


@composite
def healpixes(draw: DrawFn, order: Literal["ring", "nested"] = "ring") -> hp.HEALPix:
    nside = draw(integers(min_value=MIN_NSIDE_EXP, max_value=MAX_NSIDE_EXP).map(partial(pow, 2)))
    return hp.HEALPix(nside=nside, order=order)


@composite
def frames(draw: DrawFn) -> type[coords.BaseCoordinateFrame]:
    return draw(
        sampled_from(
            [
                coords.BarycentricTrueEcliptic,
                coords.ICRS,
                coords.Galactic,
                "galactic",
                "barycentrictrueecliptic",
                "icrs",
            ]
        )
    )


@composite
def coords_in(draw: DrawFn) -> str:
    return draw(sampled_from(["E", "G", "C"]))


@composite
def sky_coords(draw: DrawFn) -> coords.SkyCoord:
    theta_strategy = floats(min_value=0, max_value=360)
    phi_strategy = floats(min_value=-90, max_value=90)
    obs_time = draw(times())
    shape = draw(integers(min_value=1, max_value=MAX_ANGELS_LEN))

    theta_array_strategy = arrays(dtype=float, shape=shape, elements=theta_strategy).map(
        partial(u.Quantity, unit=u.deg)
    )
    phi_array_strategy = arrays(dtype=float, shape=shape, elements=phi_strategy).map(
        partial(u.Quantity, unit=u.deg)
    )
    frame = draw(frames())
    lon = draw(theta_array_strategy)
    lat = draw(phi_array_strategy)
    return coords.SkyCoord(lon, lat, frame=frame, obstime=obs_time)


@composite
def pixels(draw: DrawFn, nside: int) -> int | list[int] | npt.NDArray[np.integer]:
    healpix = hp.HEALPix(nside=nside)
    pixel_strategy = integers(min_value=0, max_value=healpix.npix - 1)

    shape = draw(integers(min_value=1, max_value=healpix.npix - 1))

    list_stategy = lists(pixel_strategy, min_size=1)
    array_strategy = arrays(dtype=int, shape=shape, elements=pixel_strategy)

    return draw(one_of(pixel_strategy, list_stategy, array_strategy))


@composite
def angles(draw: DrawFn, lonlat: bool = False) -> tuple[u.Quantity[u.deg], u.Quantity[u.deg]]:
    if lonlat:
        theta_strategy = floats(min_value=0, max_value=360)
        phi_strategy = floats(min_value=-90, max_value=90)
    else:
        theta_strategy = floats(min_value=0, max_value=180)
        phi_strategy = floats(min_value=0, max_value=360)

    shape = draw(integers(min_value=1, max_value=MAX_ANGELS_LEN))

    theta_array_strategy = arrays(dtype=float, shape=shape, elements=theta_strategy).map(
        partial(u.Quantity, unit=u.deg)
    )
    phi_array_strategy = arrays(dtype=float, shape=shape, elements=phi_strategy).map(
        partial(u.Quantity, unit=u.deg)
    )

    return draw(theta_array_strategy), draw(phi_array_strategy)


@composite
def freqs(
    draw: DrawFn,
    min_freq: u.Quantity,
    max_freq: u.Quantity,
    extrapolate: bool,
) -> u.Quantity[u.GHz] | u.Quantity[u.micron]:
    if extrapolate:
        return draw(sampled_from(FREQ_LOG_RANGE).map(np.exp).map(partial(u.Quantity, unit=u.GHz)))

    freq_range = np.geomspace(np.log(min_freq.value), np.log(max_freq.value), N_FREQS)
    freq_strategy = (
        sampled_from(freq_range.tolist()).map(np.exp).map(partial(u.Quantity, unit=min_freq.unit))
    )

    return np.clip(draw(freq_strategy), min_freq, max_freq)


@composite
def random_freqs(draw: DrawFn, unit: u.Unit | None = None, bandpass: bool = False) -> u.Quantity:
    random_freq = draw(
        sampled_from(FREQ_LOG_RANGE).map(np.exp).map(partial(u.Quantity, unit=u.GHz))
    )
    if unit is not None:
        random_freq = random_freq.to(unit, u.spectral())
    else:
        micron = draw(booleans())
        if micron:
            random_freq = random_freq.to(u.micron, u.spectral())

    if bandpass:
        shape = draw(integers(min_value=2, max_value=100))
        sigma = draw(floats(min_value=0.1, max_value=0.3))
        return np.linspace(
            random_freq - random_freq * sigma, random_freq + random_freq * sigma, shape
        )

    return random_freq


@composite
def weights(
    draw: DrawFn, freqs: u.Quantity[u.GHz] | u.Quantity[u.micron]
) -> list[float] | npt.NDArray[np.float64]:
    def normalize_array(
        array: Sequence[float] | npt.NDArray[np.float64],
        freqs: u.Quantity[u.GHz] | u.Quantity[u.micron],
    ) -> u.Quantity[u.MJy / u.sr]:
        return (array / np.trapz(array, freqs)).value

    weights_strategy = floats(min_value=1, max_value=100)
    list_stategy = lists(weights_strategy, min_size=freqs.size, max_size=freqs.size)
    array_strategy = arrays(dtype=float, shape=freqs.size, elements=weights_strategy)

    return draw(one_of(list_stategy, array_strategy).map(partial(normalize_array, freqs=freqs)))


@composite
def obs_positions(draw: DrawFn, model: zodipy.Zodipy, obs_time: time.Time) -> str:
    def get_obs_dist(obs: str, obs_time: time.Time) -> u.Quantity[u.AU]:
        if obs == "semb-l2":
            obs_pos = (
                coords.get_body("earth", obs_time)
                .transform_to(coords.HeliocentricMeanEcliptic)
                .cartesian.xyz
            )
            obs_pos += 0.01 * u.AU
        else:
            obs_pos = (
                coords.get_body(obs, obs_time)
                .transform_to(coords.HeliocentricMeanEcliptic)
                .cartesian.xyz
            )
        return u.Quantity(np.linalg.norm(obs_pos.value), u.AU)

    los_dist_cut = min(
        [COMPONENT_CUTOFFS[comp][1] for comp in model._ipd_model.comps],
    )
    if isinstance(los_dist_cut, dict):
        los_dist_cut = min(list(los_dist_cut.values()))

    obs_list = model.supported_observers
    return draw(
        sampled_from(obs_list).filter(lambda obs: los_dist_cut > get_obs_dist(obs, obs_time).value)
    )


@composite
def any_obs(draw: DrawFn, model: zodipy.Zodipy) -> str:
    return draw(sampled_from(model.supported_observers))


@composite
def zodipy_models(draw: DrawFn, **static_params: dict[str, Any]) -> zodipy.Zodipy:
    extrapolate = static_params.pop("extrapolate", draw(booleans()))
    model = static_params.pop("model", draw(sampled_from(AVAILABLE_MODELS)))
    ipd_model = model_registry.get_model(model)
    min_freq = ipd_model.spectrum.min()
    max_freq = ipd_model.spectrum.max()

    do_bp = static_params.pop("bandpass_integrate", None)
    if do_bp is not None:
        frequencies = draw(random_freqs(bandpass=True))
        w = draw(weights(frequencies))
    else:
        frequencies = static_params.pop(
            "freq", draw(freqs(min=min_freq, max=max_freq, extrapolate=extrapolate))
        )
        w = None

    gauss_quad_degree = static_params.pop(
        "gauss_quad_degree", draw(integers(min_value=1, max_value=200))
    )

    return zodipy.Zodipy(
        freq=frequencies,
        model=model,
        weights=w,
        gauss_quad_degree=gauss_quad_degree,
        extrapolate=extrapolate,
    )
