from astropy.units import Unit

from .source_parameters import SPECTRUM_DIRBE, SOLAR_IRRADIANCE_DIRBE
from ._solar_irradiance_model import solar_irradiance_model_registry


# Temporary storage location for Solar irradiance tables
DOWNLOAD_URL = "http://tsih3.uio.no/www_cmb/metins/"

solar_irradiance_model_registry.register_model_from_table(
    name="dirbe", spectrum=SPECTRUM_DIRBE, irradiance=SOLAR_IRRADIANCE_DIRBE
)

solar_irradiance_model_registry.register_model_from_url(
    name="thuillier",
    url=DOWNLOAD_URL + "thuillier2004_flux.txt",
    spectrum_unit=Unit("nm"),
    irradiance_unit=Unit("mW/(m^2 nm sr)"),
    skip_rows=2,
)

solar_irradiance_model_registry.register_model_from_url(
    name="gueymard",
    url=DOWNLOAD_URL + "gueymard2003_flux.txt",
    spectrum_unit=Unit("nm"),
    irradiance_unit=Unit("W/(m^2 nm sr)"),
    skip_rows=9,
)
