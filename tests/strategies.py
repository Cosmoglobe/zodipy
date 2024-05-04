import datetime

import numpy as np
from astropy import time, units
from astropy.coordinates import SkyCoord
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from zodipy import Model
from zodipy.model_registry import model_registry
from zodipy.zodiacal_light_model import ZodiacalLightModel

MIN_DATE = datetime.datetime(year=1900, month=1, day=1)
MAX_DATE = datetime.datetime(year=2100, month=1, day=1)


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
def obstimes(draw: st.DrawFn) -> time.Time:
    """Return a strategy for generating astropy Time objects."""
    return draw(st.datetimes(min_value=MIN_DATE, max_value=MAX_DATE).map(time.Time))


@st.composite
def obspos_xyz(draw: st.DrawFn) -> units.Quantity:
    """Return a strategy for generating a heliocentric ecliptic position."""
    positive_elements = st.floats(min_value=0.1, max_value=1)
    sign = st.sampled_from([-1, 1])
    elements = st.builds(lambda x, y: x * y, positive_elements, sign)
    vector = draw(arrays(dtype=float, shape=3, elements=elements))
    normalized_vector = vector / np.linalg.norm(vector)
    magnitude = draw(st.floats(min_value=0.8, max_value=2))
    return units.Quantity(normalized_vector * magnitude, unit=units.AU)


@st.composite
def obspos_str(draw: st.DrawFn) -> str:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(st.sampled_from(["earth", "mars", "moon", "semb-l2"]))


@st.composite
def obs(draw: st.DrawFn) -> tuple[str, units.Quantity]:
    """Return a strategy for generating a heliocentric ecliptic position."""
    return draw(st.one_of(obspos_str(), obspos_xyz()))


@st.composite
def frames(draw: st.DrawFn) -> units.Quantity:
    """Return a strategy for generating astropy coordinate frames."""
    return draw(st.sampled_from(["galactic", "barycentricmeanecliptic", "icrs"]))


@st.composite
def sky_coords(draw: st.DrawFn) -> SkyCoord:
    """Return a strategy for generating astropy SkyCoord objects."""
    theta_strategy = st.floats(min_value=0, max_value=360)
    phi_strategy = st.floats(min_value=-90, max_value=90)
    ncoords = draw(st.integers(min_value=1, max_value=1000))

    theta_array_strategy = arrays(dtype=float, shape=ncoords, elements=theta_strategy)
    phi_array_strategy = arrays(dtype=float, shape=ncoords, elements=phi_strategy)
    return SkyCoord(
        draw(theta_array_strategy),
        draw(phi_array_strategy),
        unit=(units.deg),
        frame=draw(frames()),
        obstime=draw(obstimes()),
    )
