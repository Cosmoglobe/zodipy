from hypothesis import given

from zodipy.zodipy import Zodipy

from ._strategies import model


@given(model())
def test_ipd_model_repr(model: Zodipy) -> None:
    """Tests that the IPD model has a userfriendly repr."""
    repr(model)
    repr(model._ipd_model)
