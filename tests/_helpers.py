from __future__ import annotations

import datetime
import random

import astropy.units as u
import numpy as np
from astropy.time import Time

import zodipy
from zodipy._ephemeris import get_obs_and_earth_positions

MIN_FREQ = u.Quantity(0.1, u.GHz)
MAX_FREQ = u.Quantity(500, u.micron).to(u.GHz, equivalencies=u.spectral())
MIN_DATE = datetime.datetime(year=1900, month=1, day=1)
MAX_DATE = datetime.datetime(year=2100, month=1, day=1)
MIN_NSIDE = 8
MAX_NSIDE = 1024


def get_random_model() -> str:
    random.seed()
    return random.choice(zodipy.model_registry.models)


def get_random_observer(model: zodipy.Zodipy) -> str:
    random.seed()
    return random.choice(model.supported_observers)


def update_cutoff(model: zodipy.Zodipy, obs: str, time: Time) -> None:
    obs_pos, _ = get_obs_and_earth_positions(obs, time, None)
    obs_dist = np.linalg.norm(obs_pos) * u.AU
    if model.los_cutoff < obs_dist:
        model.los_cutoff = obs_dist + 1 * u.AU
