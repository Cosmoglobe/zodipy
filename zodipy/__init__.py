import pkg_resources

from . import ipd_comps, source_params
from ._contour import tabulate_density
from .ipd_models import model_registry
from .zodipy import Zodipy

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:
    ...

__all__ = (
    "Zodipy",
    "model_registry",
    "ipd_comps",
    "source_params",
    "tabulate_density",
)
