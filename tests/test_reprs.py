from hypothesis import given

from zodipy.zodipy import Model

from ._strategies import zodipy_models


@given(zodipy_models())
def test_ipd_model_repr(model: Model) -> None:
    """Tests that the IPD model has a userfriendly repr."""
    repr(model)
    repr(model._interplanetary_dust_model)
