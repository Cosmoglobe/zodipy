import numpy as np

from zodipy._los_config import LOSFactory
from zodipy._components import ComponentLabel

EPS = np.finfo(float).eps
RADIAL_CUTOFF = 6


LOS_configs = LOSFactory()

LOS_configs.register_config(
    name="default",
    components={  # "comp name": (start, stop, n, geometry = "linear" | "log")
        ComponentLabel.CLOUD: (EPS, RADIAL_CUTOFF, 250, "linear"),
        ComponentLabel.BAND1: (EPS, RADIAL_CUTOFF, 50, "linear"),
        ComponentLabel.BAND2: (EPS, RADIAL_CUTOFF, 50, "linear"),
        ComponentLabel.BAND3: (EPS, RADIAL_CUTOFF, 50, "linear"),
        ComponentLabel.RING: (EPS, 2.25, 50, "linear"),
        ComponentLabel.FEATURE: (EPS, 1, 50, "linear"),
    },
)
LOS_configs.register_config(
    name="high",
    components={
        ComponentLabel.CLOUD: (EPS, RADIAL_CUTOFF, 500, "linear"),
        ComponentLabel.BAND1: (EPS, RADIAL_CUTOFF, 500, "linear"),
        ComponentLabel.BAND2: (EPS, RADIAL_CUTOFF, 500, "linear"),
        ComponentLabel.BAND3: (EPS, RADIAL_CUTOFF, 500, "linear"),
        ComponentLabel.RING: (EPS, 2.25, 200, "linear"),
        ComponentLabel.FEATURE: (EPS, 1, 200, "linear"),
    },
)
LOS_configs.register_config(
    name="fast",
    components={
        ComponentLabel.CLOUD: (EPS, RADIAL_CUTOFF, 25, "linear"),
        ComponentLabel.BAND1: (EPS, RADIAL_CUTOFF, 25, "linear"),
        ComponentLabel.BAND2: (EPS, RADIAL_CUTOFF, 25, "linear"),
        ComponentLabel.BAND3: (EPS, RADIAL_CUTOFF, 25, "linear"),
        ComponentLabel.RING: (EPS, 2.25, 25, "linear"),
        ComponentLabel.FEATURE: (EPS, 1, 25, "linear"),
    },
)
