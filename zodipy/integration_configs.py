from zodipy._integration_config import IntegrationConfigFactory


RADIAL_CUTOFF = 6

integration_configs = IntegrationConfigFactory()

integration_configs.register_config(
    name="default",
    components={
        "cloud": (RADIAL_CUTOFF, 250),  # (R_max, n_LOS)
        "band1": (RADIAL_CUTOFF, 50),
        "band2": (RADIAL_CUTOFF, 50),
        "band3": (RADIAL_CUTOFF, 50),
        "ring": (2.25, 50),
        "feature": (1, 50),
    },
)
integration_configs.register_config(
    name="high",
    components={
        "cloud": (RADIAL_CUTOFF, 500),
        "band1": (RADIAL_CUTOFF, 500),
        "band2": (RADIAL_CUTOFF, 500),
        "band3": (RADIAL_CUTOFF, 500),
        "ring": (2.25, 200),
        "feature": (1, 200),
    },
)
integration_configs.register_config(
    name="fast",
    components={
        "cloud": (RADIAL_CUTOFF, 25),
        "band1": (RADIAL_CUTOFF, 25),
        "band2": (RADIAL_CUTOFF, 25),
        "band3": (RADIAL_CUTOFF, 25),
        "ring": (2.25, 25),
        "feature": (1, 25),
    },
)
