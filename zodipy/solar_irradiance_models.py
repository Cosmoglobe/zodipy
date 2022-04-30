import astropy.units as u

from ._solar_irradiance_model import solar_irradiance_model_registry
from .source_parameters import SOLAR_IRRADIANCE_DIRBE, SPECTRUM_DIRBE

# Temporary storage location for Solar irradiance tables
DOWNLOAD_URL = "http://tsih3.uio.no/www_cmb/metins/"

solar_irradiance_model_registry.register_model_from_table(
    name="dirbe", spectrum=SPECTRUM_DIRBE, irradiance=SOLAR_IRRADIANCE_DIRBE
)

solar_irradiance_model_registry.register_model_from_url(
    name="thuillier",
    url=DOWNLOAD_URL + "thuillier2004_flux.txt",
    spectrum_unit=u.Unit("nm"),
    irradiance_unit=u.Unit("mW/(m^2 nm sr)"),
    skip_rows=2,
)

solar_irradiance_model_registry.register_model_from_url(
    name="gueymard",
    url=DOWNLOAD_URL + "gueymard2003_flux.txt",
    spectrum_unit=u.Unit("nm"),
    irradiance_unit=u.Unit("W/(m^2 nm sr)"),
    skip_rows=9,
)
