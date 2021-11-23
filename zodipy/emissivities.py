from zodipy._component_labels import ComponentLabel
from zodipy._emissivities import Emissivity


FREQUENCIES_PLANCK = (100.0, 143.0, 217.0, 353.0, 545.0, 857.0)

PLANCK_2013 = Emissivity(
    frequencies=FREQUENCIES_PLANCK,
    components={
        ComponentLabel.CLOUD: (0.003, -0.014, 0.031, 0.168, 0.223, 0.301),
        ComponentLabel.BAND1: (1.129, 1.463, 2.024, 2.035, 2.235, 1.777),
        ComponentLabel.BAND2: (0.674, 0.530, 0.338, 0.436, 0.718, 0.716),
        ComponentLabel.BAND3: (1.106, 1.794, 2.507, 2.400, 3.193, 2.870),
        ComponentLabel.RING: (0.163, -0.252, -0.185, -0.211, 0.591, 0.578),
        ComponentLabel.FEATURE: (0.252, -0.002, 0.243, 0.676, -0.182, 0.423),
    },
)
PLANCK_2015 = Emissivity(
    frequencies=FREQUENCIES_PLANCK,
    components={
        ComponentLabel.CLOUD: (0.012, 0.022, 0.051, 0.106, 0.167, 0.256),
        ComponentLabel.BAND1: (1.02, 1.23, 1.30, 1.58, 1.74, 2.06),
        ComponentLabel.BAND2: (0.08, 0.15, 0.15, 0.39, 0.54, 0.85),
        ComponentLabel.BAND3: (0.72, 1.16, 1.27, 1.88, 2.54, 3.37),
    },
)
PLANCK_2018 = Emissivity(
    frequencies=FREQUENCIES_PLANCK,
    components={
        ComponentLabel.CLOUD: (0.018, 0.020, 0.042, 0.082, 0.179, 0.304),
        ComponentLabel.BAND1: (0.54, 1.00, 1.11, 1.52, 1.47, 1.58),
        ComponentLabel.BAND2: (0.07, 0.17, 0.21, 0.35, 0.49, 0.70),
        ComponentLabel.BAND3: (0.19, 0.84, 1.12, 1.77, 1.84, 2.11),
    },
)
