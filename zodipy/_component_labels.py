from enum import Enum
from typing import Dict, Type

from zodipy._components import Component, Cloud, Band, Ring, Feature


class ComponentLabel(Enum):
    """Labels representing the six Zodiacal components in the K98 IPD model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"


LABEL_TO_CLASS: Dict[ComponentLabel, Type[Component]] = {
    ComponentLabel.CLOUD: Cloud,
    ComponentLabel.BAND1: Band,
    ComponentLabel.BAND2: Band,
    ComponentLabel.BAND3: Band,
    ComponentLabel.RING: Ring,
    ComponentLabel.FEATURE: Feature,
}
