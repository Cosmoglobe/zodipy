import pytest

import astropy.units as u
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from zodipy._brightness_integral import brightness_integral
from zodipy.models import model_registry
from zodipy._labels import Label

NSIDE = 128
NPIX = hp.nside2npix(NSIDE)
FREQ = (25*u.micron).to("GHz", equivalencies=u.spectral()).value
MODEL = model_registry.get_model("DIRBE")
COMPONENT = Label.CLOUD
LOS = np.linspace(0.001, 10, 50)
OBSERVER_POS = np.array([1.0, 0.0, 0.0])
UNIT_VECTORS = np.asarray(hp.pix2vec(NSIDE, np.arange(NPIX)))

def test_brightness_integral_shape():

    emission = brightness_integral(
        freq=FREQ,
        model=MODEL,
        component_label=COMPONENT,
        radial_distances=LOS,
        observer_pos=OBSERVER_POS,
        earth_pos=OBSERVER_POS,
        unit_vectors=UNIT_VECTORS,
        color_table=None,
    )

    assert emission.shape == (NPIX,)

    emission = brightness_integral(
        freq=FREQ,
        model=MODEL,
        component_label=COMPONENT,
        radial_distances=LOS,
        observer_pos=OBSERVER_POS,
        earth_pos=OBSERVER_POS,
        unit_vectors=UNIT_VECTORS[:5000],
        color_table=None,
    )

    assert emission.shape == np.shape(UNIT_VECTORS[:5000][-1])