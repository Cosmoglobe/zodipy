import pytest

from zodipy import Zodipy


@pytest.fixture
def DIRBE():
    return Zodipy(model="dirbe")


@pytest.fixture
def PLANCK18():
    return Zodipy(model="planck18")
