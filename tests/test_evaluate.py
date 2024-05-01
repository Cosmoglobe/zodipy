import pytest
from astropy import coordinates as coords
from astropy import time, units

from zodipy import Model

from .dirbe_tabulated import DAYS, DIRBE_START_DAY, LAT, LON, TABULATED_DIRBE_EMISSION


def test_compare_to_dirbe_idl() -> None:
    """Tests that ZodiPy reproduces the DIRBE software.

    Zodipy should be able to reproduce the tabulated emission from the DIRBE Zoidacal Light
    Prediction Software with a maximum difference of 0.1%.
    """
    for frequency, tabulated_emission in TABULATED_DIRBE_EMISSION.items():
        model = Model(x=frequency * units.micron, name="dirbe")
        for idx, (day, lon, lat) in enumerate(zip(DAYS, LON, LAT)):
            obstime = DIRBE_START_DAY + time.TimeDelta(day - 1, format="jd")
            coord = coords.SkyCoord(
                lon,
                lat,
                unit=units.deg,
                frame=coords.BarycentricMeanEcliptic,
                obstime=obstime,
            )
            emission = model.evaluate(coord)
            assert emission.value == pytest.approx(tabulated_emission[idx], rel=0.01)
