from __future__ import annotations

import datetime
import random
from math import log2
from typing import Callable

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.time import Time
from hypothesis.extra import numpy
from hypothesis.strategies import (
    DrawFn,
    SearchStrategy,
    booleans,
    composite,
    datetimes,
    floats,
    integers,
)
from numpy.typing import NDArray

import zodipy

from ._helpers import get_random_model

MIN_FREQ = u.Quantity(0.1, u.GHz)
MAX_FREQ = u.Quantity(500, u.micron).to(u.GHz, equivalencies=u.spectral())
MIN_DATE = datetime.datetime(year=1900, month=1, day=1)
MAX_DATE = datetime.datetime(year=2100, month=1, day=1)
MIN_NSIDE = 8
MAX_NSIDE = 1024


@composite
def angles(draw: DrawFn) -> tuple[u.Quantity[u.deg], u.Quantity[u.deg]]:
    angles_shapes = numpy.array_shapes(
        min_dims=1, max_dims=1, min_side=1, max_side=10000
    )
    shape = draw(angles_shapes)
    theta = numpy.arrays(
        dtype=float,
        shape=shape,
        elements=floats(min_value=0, max_value=180),
    )
    phi = numpy.arrays(
        dtype=float,
        shape=shape,
        elements=floats(min_value=0, max_value=360),
    )
    return u.Quantity(draw(theta), u.deg), u.Quantity(draw(phi), u.deg)


@composite
def time(
    draw: Callable[[SearchStrategy[datetime.datetime]], datetime.datetime]
) -> Time:
    datetime = draw(datetimes(min_value=MIN_DATE, max_value=MAX_DATE))
    return Time(datetime)


@composite
def nside(draw: Callable[[SearchStrategy[int]], int]) -> int:
    power = draw(
        integers(min_value=int(log2(MIN_NSIDE)), max_value=int(log2(MAX_NSIDE)))
    )
    return 2**power


@composite
def model(
    draw: DrawFn,
    model_name: str | None = None,
    quad_points: int | None = None,
    extrapolate: bool | None = None,
    los_cutoff: u.Quantity[u.AU] | None = None,
    solar_cutoff: u.Quantity[u.deg] | None = None,
) -> zodipy.Zodipy:
    if model_name is None:
        model_name = get_random_model()
    if quad_points is None:
        quad_points = draw(integers(min_value=1, max_value=200))
    if extrapolate is None:
        extrapolate = draw(booleans())
    if los_cutoff is None:
        los_cutoff_draw = draw(floats(min_value=3, max_value=100))
        los_cutoff = u.Quantity(los_cutoff_draw, u.AU)
    if solar_cutoff is None:
        solar_cutoff_draw = draw(floats(min_value=0, max_value=360))
        solar_cutoff = u.Quantity(solar_cutoff_draw, u.deg)

    return zodipy.Zodipy(
        model_name,
        gauss_quad_order=quad_points,
        extrapolate=extrapolate,
        los_cutoff=los_cutoff,
        solar_cutoff=solar_cutoff,
    )


@composite
def pixels_strategy(
    draw: Callable[[SearchStrategy[NDArray[np.integer]]], NDArray[np.integer]],
    nside: int,
) -> NDArray[np.integer]:
    pixel_shapes = numpy.array_shapes(
        min_dims=1, max_dims=1, min_side=1, max_side=10000
    )
    npix = hp.nside2npix(nside) - 1
    pixels = numpy.arrays(
        dtype=int, shape=pixel_shapes, elements=integers(min_value=0, max_value=npix)
    )
    return draw(pixels)


@composite
def freq_strategy(
    draw: Callable[[SearchStrategy[float]], float], model: zodipy.Zodipy
) -> u.Quantity[u.GHz]:
    if model.extrapolate:
        freq = draw(floats(min_value=MIN_FREQ.value, max_value=MAX_FREQ.value))
    else:
        a, b = model.model.spectrum.value[0], model.model.spectrum.value[-1]
        freq = random.uniform(a, b)

    return u.Quantity(freq, model.model.spectrum.unit)
