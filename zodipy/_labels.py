from enum import Enum
from typing import Dict, Type

from zodipy._components import Component, Cloud, Band, Ring, Feature


class Label(Enum):
    """Labels representing the six Zodiacal components in the K98 model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"


LABEL_TO_CLASS: Dict[Label, Type[Component]] = {
    Label.CLOUD: Cloud,
    Label.BAND1: Band,
    Label.BAND2: Band,
    Label.BAND3: Band,
    Label.RING: Ring,
    Label.FEATURE: Feature,
}
