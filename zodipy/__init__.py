import pkg_resources

from zodipy import comps, source_params
from zodipy._contour import tabulate_density
from zodipy.model_registry import model_registry
from zodipy.zodipy import Zodipy

try:
    __version__ = pkg_resources.get_distribution(__name__).version
except pkg_resources.DistributionNotFound:  # pragma: no cover
    ...

__all__ = (
    "Zodipy",
    "model_registry",
    "comps",
    "source_params",
    "tabulate_density",
)
