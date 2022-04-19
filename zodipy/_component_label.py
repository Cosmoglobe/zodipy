from enum import Enum


class ComponentLabel(Enum):
    """Labels representing the components in the DIRBE model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"
