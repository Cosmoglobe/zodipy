import astropy.units as u

from zodipy.component import ComponentLabel

T_0_DIRBE = 286
DELTA_DIRBE = 0.46686259861486573
T_LARGE_GRAINS = 255
T_SMALL_GRAINS = 305
DELTA_LARGE_GRAINS = 0.5
DELTA_SMALL_GRAINS = 0.4

T_0_RRM = {
    ComponentLabel.FAN: T_LARGE_GRAINS,
    ComponentLabel.COMET: T_LARGE_GRAINS,
    ComponentLabel.INNER_NARROW_BAND: T_LARGE_GRAINS,
    ComponentLabel.OUTER_NARROW_BAND: T_LARGE_GRAINS,
    ComponentLabel.RING_RRM: T_LARGE_GRAINS,
    ComponentLabel.FEATURE_RRM: T_LARGE_GRAINS,
    ComponentLabel.BROAD_BAND: T_LARGE_GRAINS,
    ComponentLabel.INTERSTELLAR: T_SMALL_GRAINS,
}
DELTA_RMM = {
    ComponentLabel.FAN: DELTA_LARGE_GRAINS,
    ComponentLabel.COMET: DELTA_LARGE_GRAINS,
    ComponentLabel.INNER_NARROW_BAND: DELTA_LARGE_GRAINS,
    ComponentLabel.OUTER_NARROW_BAND: DELTA_LARGE_GRAINS,
    ComponentLabel.RING_RRM: DELTA_LARGE_GRAINS,
    ComponentLabel.FEATURE_RRM: DELTA_LARGE_GRAINS,
    ComponentLabel.BROAD_BAND: DELTA_LARGE_GRAINS,
    ComponentLabel.INTERSTELLAR: DELTA_SMALL_GRAINS,
}

SPECTRUM_PLANCK = u.Quantity([100.0, 143.0, 217.0, 353.0, 545.0, 857.0], u.GHz)
SPECTRUM_DIRBE = u.Quantity([1.25, 2.2, 3.5, 4.9, 12, 25, 60, 100, 140, 240], u.micron)
SPECTRUM_IRAS = u.Quantity([12, 25, 60, 100], u.micron)

#! TODO: Figure out calibration and source evaluation for rrm-experimental
# CALIBRATION_RRM = (2.45, 2.42, 2.24, 1.97)
CALIBRATION_RRM = (1, 1, 1, 1)
OFFSET_RRM = (0.48, -1.32, 0.13, -1.47)

EMISSIVITY_PLANCK_13 = {
    ComponentLabel.CLOUD: (0.003, -0.014, 0.031, 0.168, 0.223, 0.301),
    ComponentLabel.BAND1: (1.129, 1.463, 2.024, 2.035, 2.235, 1.777),
    ComponentLabel.BAND2: (0.674, 0.530, 0.338, 0.436, 0.718, 0.716),
    ComponentLabel.BAND3: (1.106, 1.794, 2.507, 2.400, 3.193, 2.870),
    ComponentLabel.RING: (0.163, -0.252, -0.185, -0.211, 0.591, 0.578),
    ComponentLabel.FEATURE: (0.252, -0.002, 0.243, 0.676, -0.182, 0.423),
}

EMISSIVITY_PLANCK_15 = {
    ComponentLabel.CLOUD: (0.012, 0.022, 0.051, 0.106, 0.167, 0.256),
    ComponentLabel.BAND1: (1.02, 1.23, 1.30, 1.58, 1.74, 2.06),
    ComponentLabel.BAND2: (0.08, 0.15, 0.15, 0.39, 0.54, 0.85),
    ComponentLabel.BAND3: (0.72, 1.16, 1.27, 1.88, 2.54, 3.37),
}


EMISSIVITY_PLANCK_18 = {
    ComponentLabel.CLOUD: (0.018, 0.020, 0.042, 0.082, 0.179, 0.304),
    ComponentLabel.BAND1: (0.54, 1.00, 1.11, 1.52, 1.47, 1.58),
    ComponentLabel.BAND2: (0.07, 0.17, 0.21, 0.35, 0.49, 0.70),
    ComponentLabel.BAND3: (0.19, 0.84, 1.12, 1.77, 1.84, 2.11),
}

EMISSIVITY_ODEGARD = {
    ComponentLabel.CLOUD: (0.014, 0.023, 0.063, 0.132, 0.210, 0.285),
    ComponentLabel.BAND1: (1.25, 1.39, 1.85, 2.41, 2.81, 3.23),
    ComponentLabel.BAND2: (0.15, 0.22, 0.40, 0.80, 1.11, 1.58),
    ComponentLabel.BAND3: (0.50, 0.89, 1.22, 1.96, 2.81, 3.60),
}

EMISSIVITY_DIRBE = {
    ComponentLabel.CLOUD: (
        1.0,
        1.0,
        1.6598924040649741,
        0.99740908486652979,
        0.95766914805948866,
        1.0,
        0.73338832616768868,
        0.64789881802224070,
        0.67694205881047387,
        0.51912085401950736,
    ),
    ComponentLabel.BAND1: (
        1.0,
        1.0,
        1.6598924040649741,
        0.35926451958350442,
        1.0127926948497732,
        1.0,
        1.2539242027824944,
        1.5167023376593836,
        1.1317240279481993,
        1.3996145963796358,
    ),
    ComponentLabel.BAND2: (
        1.0,
        1.0,
        1.6598924040649741,
        0.35926451958350442,
        1.0127926948497732,
        1.0,
        1.2539242027824944,
        1.5167023376593836,
        1.1317240279481993,
        1.3996145963796358,
    ),
    ComponentLabel.BAND3: (
        1.0,
        1.0,
        1.6598924040649741,
        0.35926451958350442,
        1.0127926948497732,
        1.0,
        1.2539242027824944,
        1.5167023376593836,
        1.1317240279481993,
        1.3996145963796358,
    ),
    ComponentLabel.RING: (
        1.0,
        1.0,
        1.6598924040649741,
        1.0675116768340536,
        1.0608768682182081,
        1.0,
        0.87266361378785184,
        1.0985346556794289,
        1.1515825707787077,
        0.85763800994217443,
    ),
    ComponentLabel.FEATURE: (
        1.0,
        1.0,
        1.6598924040649741,
        1.0675116768340536,
        1.0608768682182081,
        1.0,
        0.87266361378785184,
        1.0985346556794289,
        1.1515825707787077,
        0.85763800994217443,
    ),
}

ALBEDO_DIRBE = {
    ComponentLabel.CLOUD: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ComponentLabel.BAND1: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ComponentLabel.BAND2: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ComponentLabel.BAND3: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ComponentLabel.RING: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    ComponentLabel.FEATURE: (
        0.20411939612669797,
        0.25521132892052301,
        0.21043660481632315,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
}

C1_DIRBE = [
    -0.94209999,
    -0.52670002,
    -0.4312,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
C2_DIRBE = [
    0.1214,
    0.18719999,
    0.1715,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
C3_DIRBE = [
    -0.1648,
    -0.59829998,
    -0.63330001,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

SOLAR_IRRADIANCE_DIRBE = [
    2.3405606e8,
    1.2309874e8,
    64292872,
    35733824,
    5763843.0,
    1327989.4,
    230553.73,
    82999.336,
    42346.605,
    14409.608,
]
