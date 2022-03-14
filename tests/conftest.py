import pytest
from zodipy import Zodipy

@pytest.fixture
def DIRBE():
    return Zodipy(model="DIRBE")

@pytest.fixture
def PLANCK18():
    return Zodipy(model="Planck18")