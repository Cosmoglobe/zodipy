import pytest

import numpy as np

from zodipy._functions import blackbody_emission, interplanetary_temperature


TEMPERATURE = 30
TEMPERATURE_ARRAY = np.array([31,45,53])
R = 3
R_ARRAY = np.array([4, 5.3, 6])
DELTA = 0.324
FREQUENCY = 549


def test_blackbody_emission_value():
    """Tests that return value."""

    emission = blackbody_emission(TEMPERATURE, FREQUENCY)
    assert emission == pytest.approx(1.73442848898e-15, abs=1e-20)


def test_blackbody_emission_value_array():
    """Tests the return value given a temperature array."""

    emission = blackbody_emission(TEMPERATURE_ARRAY, FREQUENCY)
    true_values = np.array([1.82147825366e-15, 3.06550295038e-15, 3.78860400626e-15])
    assert emission == pytest.approx(true_values, abs=1e-20) 


def test_blackbody_emission_returns_float():
    """Tests that the returned value is a float given a float temperature."""

    emission = blackbody_emission(TEMPERATURE, FREQUENCY)
    assert isinstance(emission, float)


def test_blackbody_emission_returns_array():
    """Tests that the returned value is an array given an array temperature."""

    emission = blackbody_emission(TEMPERATURE_ARRAY, FREQUENCY)
    assert isinstance(emission, np.ndarray)


def test_interplanetary_temperature_value():
    """Tests that the returned value given a float R."""

    ipd_temperature = interplanetary_temperature(R, TEMPERATURE, DELTA)
    assert ipd_temperature == pytest.approx(21.0152213243, abs=1e-10)


def test_interplanetary_temperature_value_array():
    """Tests that the returned value given a float R."""

    ipd_temperature = interplanetary_temperature(R_ARRAY, TEMPERATURE, DELTA)
    true_values = np.array([19.1449315324, 17.4765568067, 16.7880498296])
    assert ipd_temperature == pytest.approx(true_values, abs=1e-10)


def test_interplanetary_temperature_returns_float():
    """Tests that the returned value is a float given a float R."""

    ipd_temperature = interplanetary_temperature(R, TEMPERATURE, DELTA)
    assert isinstance(ipd_temperature, float)


def test_interplanetary_temperature_returns_array():
    """Tests that the returned value is a array given a array R."""

    ipd_temperature = interplanetary_temperature(R_ARRAY, TEMPERATURE, DELTA)
    assert isinstance(ipd_temperature, np.ndarray)