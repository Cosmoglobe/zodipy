import astropy.units as u

from zodipy._labels import Label
from zodipy._source_parameters import SourceParameter

delta_K98 = 0.4668626
T_0_K98 = 286

SPECTRUM_PLANCK = [100.0, 143.0, 217.0, 353.0, 545.0, 857.0] * u.GHz
SPECTRUM_DIRBE = [1.25, 2.2, 3.5, 4.9, 12, 25, 60, 100, 140, 240] * u.micron

EMISSIVITY_PLANCK_13 = SourceParameter(
    values={
        Label.CLOUD: (0.003, -0.014, 0.031, 0.168, 0.223, 0.301),
        Label.BAND1: (1.129, 1.463, 2.024, 2.035, 2.235, 1.777),
        Label.BAND2: (0.674, 0.530, 0.338, 0.436, 0.718, 0.716),
        Label.BAND3: (1.106, 1.794, 2.507, 2.400, 3.193, 2.870),
        Label.RING: (0.163, -0.252, -0.185, -0.211, 0.591, 0.578),
        Label.FEATURE: (0.252, -0.002, 0.243, 0.676, -0.182, 0.423),
    },
    spectrum=SPECTRUM_PLANCK,
    doc="Emissivities fitted to the HFI Bands in the 2013 Planck analysis.",
)

EMISSIVITY_PLANCK_15 = SourceParameter(
    values={
        Label.CLOUD: (0.012, 0.022, 0.051, 0.106, 0.167, 0.256),
        Label.BAND1: (1.02, 1.23, 1.30, 1.58, 1.74, 2.06),
        Label.BAND2: (0.08, 0.15, 0.15, 0.39, 0.54, 0.85),
        Label.BAND3: (0.72, 1.16, 1.27, 1.88, 2.54, 3.37),
    },
    spectrum=SPECTRUM_PLANCK,
    doc="Emissivities fitted to the HFI Bands in the 2015 Planck analysis.",
)


EMISSIVITY_PLANCK_18 = SourceParameter(
    values={
        Label.CLOUD: (0.018, 0.020, 0.042, 0.082, 0.179, 0.304),
        Label.BAND1: (0.54, 1.00, 1.11, 1.52, 1.47, 1.58),
        Label.BAND2: (0.07, 0.17, 0.21, 0.35, 0.49, 0.70),
        Label.BAND3: (0.19, 0.84, 1.12, 1.77, 1.84, 2.11),
    },
    spectrum=SPECTRUM_PLANCK,
    doc="Emissivities fitted to the HFI Bands in the 2018 Planck analysis.",
)

EMISSIVITY_DIRBE = SourceParameter(
    values={
        Label.CLOUD: (1.0, 1.0, 1.66, 0.997, 0.958, 1.00, 0.733, 0.647, 0.677, 0.519),
        Label.BAND1: (1.0, 1.0, 1.66, 0.359, 1.01, 1.00, 1.25, 1.52, 1.13, 1.40),
        Label.BAND2: (1.0, 1.0, 1.66, 0.359, 1.01, 1.00, 1.25, 1.52, 1.13, 1.40),
        Label.BAND3: (1.0, 1.0, 1.66, 0.359, 1.01, 1.00, 1.25, 1.52, 1.13, 1.40),
        Label.RING: (1.0, 1.0, 1.66, 1.06, 1.06, 1.0, 0.873, 1.1, 1.5, 0.858),
        Label.FEATURE: (1.0, 1.0, 1.66, 1.06, 1.06, 1.0, 0.873, 1.1, 1.5, 0.858),
    },
    spectrum=SPECTRUM_PLANCK,
    doc="Emissivities fitted to ten DIRBE bands.",
)

ALBEDO_DIRBE = SourceParameter(
    values={
        Label.CLOUD: (0.204, 0.255, 0.210),
        Label.BAND1: (0.204, 0.255, 0.210),
        Label.BAND2: (0.204, 0.255, 0.210),
        Label.BAND3: (0.204, 0.255, 0.210),
        Label.RING: (0.204, 0.255, 0.210),
        Label.FEATURE: (0.204, 0.255, 0.210),
    },
    spectrum=SPECTRUM_DIRBE[:3],
    doc="Albedos fitted to first three DIRBE bands.",
)
