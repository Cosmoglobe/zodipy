import numpy as np
import pytest

from zodipy._source_funcs import get_blackbody_emission

TEMPERATURE = 30
TEMPERATURE_ARRAY = np.array([31, 45, 53])
R = 3
R_ARRAY = np.array([4, 5.3, 6])
DELTA = 0.324
FREQUENCY = 549e9


def test_blackbody_emission_value():
    """Tests that return value."""
    emission = get_blackbody_emission(T=TEMPERATURE, freq=FREQUENCY)
    assert emission == pytest.approx(1.73442848898e-15, abs=1e-20)


def test_blackbody_emission_value_array():
    """Tests the return value given a temperature array."""

    emission = get_blackbody_emission(T=TEMPERATURE_ARRAY, freq=FREQUENCY)
    true_values = np.array([1.82147825366e-15, 3.06550295038e-15, 3.78860400626e-15])
    assert emission == pytest.approx(true_values, abs=1e-20)
