from zodipy import CompLabel


_PLANCK_FREQUENCIES = (100, 143, 217, 353, 545, 857)

PLANCK_2013 = {
    'freqs' : _PLANCK_FREQUENCIES,
    CompLabel.CLOUD : (0.003, -0.014, 0.031, 0.168, 0.223, 0.301),
    CompLabel.BAND1 : (1.129, 1.463, 2.024, 2.035, 2.235, 1.777),
    CompLabel.BAND2 : (0.674, 0.530, 0.338, 0.436, 0.718, 0.716),
    CompLabel.BAND3 : (1.106, 1.794, 2.507, 2.400, 3.193, 2.870),
    CompLabel.RING : (0.163, -0.252, -0.185, -0.211, 0.591, 0.578),
    CompLabel.FEATURE : (0.252, -0.002, 0.243, 0.676, -0.182, 0.423)
}

PLANCK_2015 = {
    'freqs' : _PLANCK_FREQUENCIES,
    CompLabel.CLOUD : (0.012, 0.022, 0.051, 0.106, 0.167, 0.256),
    CompLabel.BAND1 : (1.02, 1.23, 1.30, 1.58, 1.74, 2.06),
    CompLabel.BAND2 : (0.08, 0.15, 0.15, 0.39, 0.54, 0.85),
    CompLabel.BAND3 : (0.72, 1.16, 1.27, 1.88, 2.54, 3.37),
}

PLANCK_2018 = {
    'freqs' : _PLANCK_FREQUENCIES,
    CompLabel.CLOUD : (0.018, 0.020, 0.042, 0.082, 0.179, 0.304),
    CompLabel.BAND1 : (0.54, 1.00, 1.11, 1.52, 1.47, 1.58),
    CompLabel.BAND2 : (0.07, 0.17, 0.21, 0.35, 0.49, 0.70),
    CompLabel.BAND3 : (0.19, 0.84, 1.12, 1.77, 1.84, 2.11),
}