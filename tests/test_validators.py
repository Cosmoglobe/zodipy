import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis.strategies import DataObject, data

from zodipy._types import FrequencyOrWavelength
from zodipy._validators import validate_and_normalize_weights, validate_frequencies
from zodipy.zodipy import Zodipy

from ._strategies import model, random_freq, weights

BANDPASS_FREQUENCIES = np.linspace(95, 105, 11) * u.GHz
BANDPASS_WAVELENGTHS = np.linspace(20, 25, 11) * u.micron
BANDPASS_WEIGHTS = np.array([2, 3, 5, 9, 11, 12, 11, 9, 5, 3, 2])


@given(model(extrapolate=False))
def test_validate_frequencies(model: Zodipy) -> None:
    with pytest.raises(TypeError):
        validate_frequencies(
            freq=BANDPASS_FREQUENCIES.value,
            model=model.ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(TypeError):
        validate_frequencies(
            freq=25,
            model=model.ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(u.UnitsError):
        validate_frequencies(
            freq=BANDPASS_FREQUENCIES.value * u.g,
            model=model.ipd_model,
            extrapolate=model.extrapolate,
        )
    with pytest.raises(u.UnitsError):
        validate_frequencies(
            freq=25 * u.g,
            model=model.ipd_model,
            extrapolate=model.extrapolate,
        )


@given(random_freq(bandpass=True), data())
def test_validate_weights(freq: FrequencyOrWavelength, data: DataObject) -> None:
    bp_weights = data.draw(weights(freq))
    bp_weights = validate_and_normalize_weights(
        weights=bp_weights,
        freq=freq,
    )
    assert np.trapz(bp_weights, freq.value) == pytest.approx(1.0)


def test_validate_weights_numbers() -> None:
    validate_and_normalize_weights(
        weights=None,
        freq=BANDPASS_FREQUENCIES[0],
    )
    with pytest.raises(ValueError):
        validate_and_normalize_weights(
            weights=np.array([1, 2, 3]),
            freq=BANDPASS_FREQUENCIES,
        )
        validate_and_normalize_weights(
            weights=BANDPASS_WEIGHTS[:10],
            freq=BANDPASS_FREQUENCIES[0],
        )
        validate_and_normalize_weights(
            weights=None,
            freq=BANDPASS_FREQUENCIES,
        )


def test_validate_weights_strictly_increasing() -> None:
    with pytest.raises(ValueError):
        validate_and_normalize_weights(
            weights=BANDPASS_WEIGHTS,
            freq=np.flip(BANDPASS_FREQUENCIES),
        )


def test_validate_weights_shape() -> None:
    weights = validate_and_normalize_weights(
        weights=BANDPASS_WEIGHTS,
        freq=BANDPASS_FREQUENCIES,
    )
    assert weights.shape == BANDPASS_WEIGHTS.shape

    weights = validate_and_normalize_weights(
        weights=None,
        freq=BANDPASS_FREQUENCIES[0],
    )
    assert weights.size == 1
    assert weights == np.array([1.0], dtype=np.float64)
