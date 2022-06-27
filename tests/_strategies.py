from __future__ import annotations

import datetime
from functools import partial
from math import log2
from typing import Any, Callable

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
from hypothesis.extra import numpy
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    booleans,
    builds,
    composite,
    datetimes,
    floats,
    integers,
    lists,
    sampled_from,
)
from numpy.typing import NDArray

import zodipy

MIN_FREQ = u.Quantity(10, u.GHz)
MAX_FREQ = u.Quantity(0.1, u.micron).to(u.GHz, equivalencies=u.spectral())
N_FREQS = 1000
FREQ_RANGE = np.geomspace(
    np.log(MIN_FREQ.value),
    np.log(MAX_FREQ.value),
    N_FREQS,
).tolist()
MIN_DATE = datetime.datetime(year=1900, month=1, day=1)
MAX_DATE = datetime.datetime(year=2100, month=1, day=1)
MIN_NSIDE = 8
MAX_NSIDE = 1024
MAX_PIXELS_LEN = 10000
MAX_ANGELS_LEN = 10000
AVAILABLE_MODELS = zodipy.model_registry.models
OBS_TIME = Time("2020-01-01")


@composite
def quantities(
    draw: Callable[[SearchStrategy[float]], float],
    min_value: float,
    max_value: float,
    unit: u.Unit,
) -> u.Quantity:
    return u.Quantity(draw(floats(min_value=min_value, max_value=max_value)), unit)


@composite
def time(
    draw: Callable[[SearchStrategy[datetime.datetime]], datetime.datetime]
) -> Time:
    return Time(draw(datetimes(min_value=MIN_DATE, max_value=MAX_DATE)))


@composite
def nside(draw: Callable[[SearchStrategy[int]], int]) -> int:
    power = draw(
        integers(
            min_value=int(log2(MIN_NSIDE)),
            max_value=int(log2(MAX_NSIDE)),
        )
    )
    return 2**power


@composite
def pixels(draw: DrawFn, nside: int) -> list[int] | NDArray[np.integer]:
    npix = hp.nside2npix(nside)
    pixel_strategy = integers(min_value=0, max_value=npix - 1)
    shape = draw(integers(min_value=1, max_value=npix - 1))
    use_array = draw(booleans())
    if use_array:
        return draw(numpy.arrays(dtype=int, shape=shape, elements=pixel_strategy))

    return draw(lists(pixel_strategy, min_size=shape, max_size=shape))


@composite
def angles(
    draw: DrawFn, lonlat: bool = False
) -> tuple[u.Quantity[u.deg], u.Quantity[u.deg]]:
    if lonlat:
        theta_strategy = floats(min_value=0, max_value=360)
        phi_strategy = floats(min_value=-90, max_value=90)
    else:
        theta_strategy = floats(min_value=0, max_value=180)
        phi_strategy = floats(min_value=0, max_value=360)

    shape = draw(integers(min_value=1, max_value=MAX_ANGELS_LEN))
    use_array = draw(booleans())
    if use_array:
        theta = draw(numpy.arrays(dtype=float, shape=shape, elements=theta_strategy))
        phi = draw(numpy.arrays(dtype=float, shape=shape, elements=phi_strategy))
    else:
        theta = draw(lists(theta_strategy, min_size=shape, max_size=shape))
        phi = draw(lists(phi_strategy, min_size=shape, max_size=shape))

    return u.Quantity(theta, u.deg), u.Quantity(phi, u.deg)


@composite
def freq(
    draw: DrawFn, model: zodipy.Zodipy
) -> u.Quantity[u.GHz] | u.Quantity[u.micron]:

    if model.extrapolate:
        return u.Quantity(np.exp(draw(sampled_from(FREQ_RANGE))), u.GHz)

    min_freq = model.model.spectrum.value[0]
    max_freq = model.model.spectrum.value[-1]

    freq_range = np.geomspace(np.log(min_freq), np.log(max_freq), N_FREQS).tolist()
    return u.Quantity(np.exp(draw(sampled_from(freq_range))), model.model.spectrum.unit)


@composite
def obs(draw: DrawFn, model: zodipy.Zodipy) -> str:
    def get_obs_dist(obs: str) -> u.Quantity[u.AU]:
        if obs == "semb-l2":
            obs_pos = get_body_barycentric("earth", OBS_TIME).to_cartesian().xyz
        else:
            obs_pos = get_body_barycentric(obs, OBS_TIME).to_cartesian().xyz
        return u.Quantity(np.linalg.norm(obs_pos.value), u.AU)

    los_dist_cut = model.los_dist_cut
    return draw(
        sampled_from(model.supported_observers).filter(
            lambda obs: get_obs_dist(obs) < los_dist_cut
        )
    )


@composite
def model(
    draw: DrawFn,
    model: str | None = None,
    gauss_quad_order: int | None = None,
    extrapolate: bool | None = None,
    los_dist_cut: u.Quantity[u.AU] | None = None,
    solar_cut: u.Quantity[u.deg] | None = None,
) -> zodipy.Zodipy:

    strategies: dict[str, SearchStrategy[Any]] = {}
    static_params: dict[str, Any] = {}
    if model is None:
        strategies["model"] = sampled_from(AVAILABLE_MODELS)
    else:
        static_params["model"] = model
    if gauss_quad_order is None:
        strategies["gauss_quad_order"] = integers(min_value=1, max_value=200)
    else:
        static_params["gauss_quad_order"] = gauss_quad_order
    if extrapolate is None:
        strategies["extrapolate"] = booleans()
    else:
        static_params["extrapolate"] = extrapolate
    if los_dist_cut is None:
        strategies["los_dist_cut"] = quantities(min_value=3, max_value=100, unit=u.AU)
    else:
        static_params["los_dist_cut"] = los_dist_cut
    if solar_cut is None:
        strategies["solar_cut"] = quantities(min_value=0, max_value=360, unit=u.deg)
    else:
        static_params["solar_cut"] = solar_cut

    return draw(builds(partial(zodipy.Zodipy, **static_params), **strategies))
