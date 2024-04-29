import numpy as np
import pytest
from astropy import time, units
from hypothesis import given
from hypothesis.strategies import DataObject, data

from zodipy._validators import get_validated_and_normalized_weights, get_validated_freq
from zodipy.zodipy import Zodipy

from ._strategies import random_freqs, weights, zodipy_models

BANDPASS_FREQUENCIES = np.linspace(95, 105, 11) * units.GHz
BANDPASS_WAVELENGTHS = np.linspace(20, 25, 11) * units.micron
BANDPASS_WEIGHTS = np.array([2, 3, 5, 9, 11, 12, 11, 9, 5, 3, 2])
OBS_TIME = time.Time("2021-01-01T00:00:00")


@given(zodipy_models(extrapolate=False))
def test_validate_frequencies(model: Zodipy) -> None:
    """Tests that the frequencies are validated."""
    with pytest.raises(TypeError):
        get_validated_freq(
            freq=BANDPASS_FREQUENCIES.value,
            model=model._ipd_model,
            extrapolate=False,
        )
    with pytest.raises(TypeError):
        get_validated_freq(
            freq=25,
            model=model._ipd_model,
            extrapolate=False,
        )
    with pytest.raises(units.UnitsError):
        get_validated_freq(
            freq=BANDPASS_FREQUENCIES.value * units.g,
            model=model._ipd_model,
            extrapolate=False,
        )
    with pytest.raises(units.UnitsError):
        get_validated_freq(
            freq=25 * units.g,
            model=model._ipd_model,
            extrapolate=False,
        )


@given(random_freqs(bandpass=True), data())
def test_validate_weights(freq: units.Quantity, data: DataObject) -> None:
    """Tests that the bandpass weights are validated."""
    bp_weights = data.draw(weights(freq))
    bp_weights = get_validated_and_normalized_weights(
        weights=bp_weights,
        freq=freq,
    )
    assert np.trapz(bp_weights, freq.value) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=None,
            freq=BANDPASS_FREQUENCIES,
        )


def test_validate_weights_numbers() -> None:
    """Tests that the bandpass weights are normalized."""
    get_validated_and_normalized_weights(
        weights=None,
        freq=BANDPASS_FREQUENCIES[0],
    )
    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=np.array([1, 2, 3]),
            freq=BANDPASS_FREQUENCIES,
        )
    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=BANDPASS_WEIGHTS[:10],
            freq=BANDPASS_FREQUENCIES[0],
        )
    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=None,
            freq=BANDPASS_FREQUENCIES,
        )


def test_validate_weights_strictly_increasing() -> None:
    """Tests that an error is raised if a bandpass is not strictly increasing."""
    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=BANDPASS_WEIGHTS,
            freq=np.flip(BANDPASS_FREQUENCIES),
        )


def test_validate_weights_shape() -> None:
    """Tests that the bandpass weights have the correct shape."""
    weights = get_validated_and_normalized_weights(
        weights=BANDPASS_WEIGHTS,
        freq=BANDPASS_FREQUENCIES,
    )
    assert weights.shape == BANDPASS_WEIGHTS.shape

    weights = get_validated_and_normalized_weights(
        weights=None,
        freq=BANDPASS_FREQUENCIES[0],
    )
    assert weights.size == 1
    assert weights == np.array([1.0], dtype=np.float64)


def test_extrapolate_raises_error() -> None:
    """Tests that an error is correctly raised when extrapolation is not allowed."""
    with pytest.raises(ValueError):
        model = Zodipy(freq=400 * units.micron, model="dirbe")
        model.get_emission_pix([1, 4, 5], nside=32, obs_time=OBS_TIME)

    model = Zodipy(freq=400 * units.micron, model="dirbe", extrapolate=True)
    model.get_emission_pix([1, 4, 5], nside=32, obs_time=OBS_TIME)


def test_interp_kind() -> None:
    """Tests that the interpolation kind can be passed in."""
    model = Zodipy(freq=27 * units.micron, model="dirbe", interp_kind="linear")
    linear = model.get_emission_pix([1, 4, 5], nside=32, obs_time=OBS_TIME)

    model = Zodipy(freq=27 * units.micron, model="dirbe", interp_kind="quadratic")
    quadratic = model.get_emission_pix([1, 4, 5], nside=32, obs_time=OBS_TIME)

    assert not np.allclose(linear, quadratic)

    with pytest.raises(NotImplementedError):
        model = Zodipy(freq=27 * units.micron, model="dirbe", interp_kind="sdfs")
        model.get_emission_pix(pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME)


def test_wrong_frame() -> None:
    """Tests that an error is correctly raised when an incorrect frame is passed in."""
    model = Zodipy(freq=27 * units.micron)
    with pytest.raises(ValueError):
        model.get_emission_pix(
            [1, 4, 5],
            nside=32,
            obs_time=OBS_TIME,
            coord_in="not a valid frame",
        )


def test_non_quantity_ang_raises_error() -> None:
    """Tests that an error is correctly raised if the user inputed angles are not Quantities."""
    model = Zodipy(freq=27 * units.micron)
    with pytest.raises(TypeError):
        model.get_emission_ang(
            32,
            12,
            obs_time=OBS_TIME,
        )
