from zodipy._los_config import LOSFactory

import numpy as np

EPS = np.finfo(float).eps
RADIAL_CUTOFF = 6


LOS_configs = LOSFactory()

LOS_configs.register_config(
    name="default",
    components={  # "comp name": (start, stop, n, geometry = "linear" | "log")
        "cloud": (EPS, RADIAL_CUTOFF, 250, "linear"),
        "band1": (EPS, RADIAL_CUTOFF, 50, "linear"),
        "band2": (EPS, RADIAL_CUTOFF, 50, "linear"),
        "band3": (EPS, RADIAL_CUTOFF, 50, "linear"),
        "ring": (EPS, 2.25, 50, "linear"),
        "feature": (EPS, 1, 50, "linear"),
    },
)
LOS_configs.register_config(
    name="high",
    components={
        "cloud": (EPS, RADIAL_CUTOFF, 500, "linear"),
        "band1": (EPS, RADIAL_CUTOFF, 500, "linear"),
        "band2": (EPS, RADIAL_CUTOFF, 500, "linear"),
        "band3": (EPS, RADIAL_CUTOFF, 500, "linear"),
        "ring": (EPS, 2.25, 200, "linear"),
        "feature": (EPS, 1, 200, "linear"),
    },
)
LOS_configs.register_config(
    name="fast",
    components={
        "cloud": (EPS, RADIAL_CUTOFF, 25, "linear"),
        "band1": (EPS, RADIAL_CUTOFF, 25, "linear"),
        "band2": (EPS, RADIAL_CUTOFF, 25, "linear"),
        "band3": (EPS, RADIAL_CUTOFF, 25, "linear"),
        "ring": (EPS, 2.25, 25, "linear"),
        "feature": (EPS, 1, 25, "linear"),
    },
)
