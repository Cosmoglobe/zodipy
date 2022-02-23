from enum import Enum
from typing import Dict, Type

from zodipy._components import Component, Cloud, Band, Ring, Feature


class CompLabel(Enum):
    """Labels representing the six Zodiacal components in the K98 model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"


LABEL_TO_CLASS: Dict[CompLabel, Type[Component]] = {
    CompLabel.CLOUD: Cloud,
    CompLabel.BAND1: Band,
    CompLabel.BAND2: Band,
    CompLabel.BAND3: Band,
    CompLabel.RING: Ring,
    CompLabel.FEATURE: Feature,
}
