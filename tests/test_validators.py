import astropy.units as u
from astropy.time import Time
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

from zodipy._types import FrequencyOrWavelength
from zodipy._validators import get_validated_and_normalized_weights, get_validated_freq
from zodipy.zodipy import Zodipy

from ._strategies import model, random_freq, weights

BANDPASS_FREQUENCIES = np.linspace(95, 105, 11) * u.GHz
BANDPASS_WAVELENGTHS = np.linspace(20, 25, 11) * u.micron
BANDPASS_WEIGHTS = np.array([2, 3, 5, 9, 11, 12, 11, 9, 5, 3, 2])
OBS_TIME = Time("2021-01-01T00:00:00")


@given(model(extrapolate=False))
def test_validate_frequencies(model: Zodipy) -> None:
    with pytest.raises(TypeError):
        get_validated_freq(
            freq=BANDPASS_FREQUENCIES.value,
            model=model._ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(TypeError):
        get_validated_freq(
            freq=25,
            model=model._ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(u.UnitsError):
        get_validated_freq(
            freq=BANDPASS_FREQUENCIES.value * u.g,
            model=model._ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(u.UnitsError):
        get_validated_freq(
            freq=25 * u.g,
            model=model._ipd_model,
            extrapolate=model.extrapolate,
        )


@given(random_freq(bandpass=True), data())
def test_validate_weights(freq: FrequencyOrWavelength, data: DataObject) -> None:
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
    with pytest.raises(ValueError):
        get_validated_and_normalized_weights(
            weights=BANDPASS_WEIGHTS,
            freq=np.flip(BANDPASS_FREQUENCIES),
        )


def test_validate_weights_shape() -> None:
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
    with pytest.raises(ValueError):
        model = Zodipy("dirbe")
        model.get_emission_pix(
            400 * u.micron, pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME
        )

    model = Zodipy("dirbe", extrapolate=True)
    model.get_emission_pix(
        400 * u.micron, pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME
    )


def test_interp_kind() -> None:
    model = Zodipy("dirbe", interp_kind="linear")
    linear = model.get_emission_pix(
        27 * u.micron, pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME
    )

    model = Zodipy("dirbe", interp_kind="quadratic")
    quadratic = model.get_emission_pix(
        27 * u.micron, pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME
    )

    assert not np.allclose(linear, quadratic)

    with pytest.raises(NotImplementedError):
        model = Zodipy("dirbe", interp_kind="sdfs")
        model.get_emission_pix(
            27 * u.micron, pixels=[1, 4, 5], nside=32, obs_time=OBS_TIME
        )
