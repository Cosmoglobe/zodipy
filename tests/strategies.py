from __future__ import annotations

import datetime

import numpy as np
import numpy.typing as npt
from astropy import time, units
from astropy.coordinates import SkyCoord
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from zodipy import Model
from zodipy.model_registry import model_registry
from zodipy.zodiacal_light_model import ZodiacalLightModel

MIN_DATE = time.Time(datetime.datetime(year=1960, month=1, day=1))
MAX_DATE = time.Time(datetime.datetime(year=2080, month=1, day=1))
TEST_SAMPRATE = 1 / 86400  # 1 sec


@st.composite
def bandpass(draw: st.DrawFn, model: ZodiacalLightModel) -> tuple[units.Quantity, np.ndarray]:
    """Generate randomized bandpass."""
    min_freq = model.spectrum.min()
    max_freq = model.spectrum.max()
    scale = draw(st.floats(min_value=0.1, max_value=0.5))
    full_width = max_freq - min_freq
    width = full_width * scale
    size = draw(st.integers(min_value=10, max_value=100))
    x = (
        np.linspace(0, width, size)
        + min_freq
        + (full_width - width) * draw(st.floats(min_value=0, max_value=1))
    )
    weights = draw(arrays(dtype=float, shape=size, elements=st.floats(min_value=0.1, max_value=1)))
    return x, weights


@st.composite
def models(draw: st.DrawFn) -> Model:
    """Generate randomized model."""
    model_name = draw(st.sampled_from(model_registry.models))
    zodi_model = model_registry.get_model(model_name)
    has_bandpass = draw(st.booleans())
    if has_bandpass:
        x, weights = draw(bandpass(zodi_model))

    else:
        x = (
            draw(
                st.floats(
                    min_value=zodi_model.spectrum.min().value,
                    max_value=zodi_model.spectrum.max().value,
                )
            )
        ) * zodi_model.spectrum.unit
        weights = None
    return Model(x, weights=weights, name=model_name)


@st.composite
def obstime_inst(draw: st.DrawFn) -> time.Time:
    """Return a strategy for generating astropy Time objects."""
    t0 = draw(st.integers(min_value=MIN_DATE.mjd, max_value=MAX_DATE.mjd))
    return time.Time(t0, format="mjd")


@st.composite
def obstime_tod(draw: st.DrawFn, size: int) -> time.Time:
    """Return a strategy for generating astropy Time objects."""
    t0 = draw(st.integers(min_value=MIN_DATE.mjd, max_value=MAX_DATE.mjd))
    return time.Time(np.linspace(t0, t0 + TEST_SAMPRATE * size, size), format="mjd")


@st.composite
def get_obspos_vec(draw: st.DrawFn, size: int) -> units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    shape = (3, size) if size != 1 else 3
    positive_elements = st.floats(min_value=0.1, max_value=1)
    sign = st.sampled_from([-1, 1])
    elements = st.builds(lambda x, y: x * y, positive_elements, sign)
    vector = draw(arrays(dtype=float, shape=shape, elements=elements))
    normalized_vector = vector / np.linalg.norm(vector, axis=0)
    magnitude = draw(st.floats(min_value=0.8, max_value=2))
    return units.Quantity(normalized_vector * magnitude, unit=units.AU)


@st.composite
def obspos_xyz_inst(draw: st.DrawFn) -> units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(get_obspos_vec(1))


@st.composite
def obspos_xyz_tod(draw: st.DrawFn, size: int) -> units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(get_obspos_vec(size))


@st.composite
def obspos_str(draw: st.DrawFn) -> str:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(st.sampled_from(["earth", "mars", "moon", "semb-l2"]))


@st.composite
def obs_inst(draw: st.DrawFn) -> str | units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(st.one_of(obspos_str(), obspos_xyz_inst()))


@st.composite
def obs_tod(draw: st.DrawFn, size: int) -> str | units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(st.one_of(obspos_str(), obspos_xyz_tod(size)))


@st.composite
def frames(draw: st.DrawFn) -> units.Quantity:
    """Return a strategy for generating astropy coordinate frames."""
    return draw(st.sampled_from(["galactic", "barycentricmeanecliptic", "icrs"]))


@st.composite
def get_lonlat(draw: st.DrawFn) -> tuple[npt.NDArray, npt.NDArray]:
    """Return a strategy for generating longitude and latitude arrays."""
    theta_strategy = st.floats(min_value=0, max_value=360)
    phi_strategy = st.floats(min_value=-90, max_value=90)
    ncoords = draw(st.integers(min_value=1, max_value=1000))

    theta_array_strategy = arrays(dtype=float, shape=ncoords, elements=theta_strategy)
    phi_array_strategy = arrays(dtype=float, shape=ncoords, elements=phi_strategy)
    return draw(theta_array_strategy), draw(phi_array_strategy)


@st.composite
def sky_coord_inst(draw: st.DrawFn) -> SkyCoord:
    """Return a strategy for generating astropy SkyCoord objects."""
    lon, lat = draw(get_lonlat())
    return SkyCoord(lon, lat, unit=(units.deg), frame=draw(frames()), obstime=draw(obstime_inst()))


@st.composite
def sky_coord_tod(draw: st.DrawFn) -> SkyCoord:
    """Return a strategy for generating astropy SkyCoord objects."""
    lon, lat = draw(get_lonlat())
    obstime = draw(obstime_tod(lon.size))
    return SkyCoord(lon, lat, unit=(units.deg), frame=draw(frames()), obstime=obstime)
