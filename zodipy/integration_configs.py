from zodipy._integration import IntegrationConfigFactory


_RADIAL_CUTOFF = 6

integration_configs = IntegrationConfigFactory()

integration_configs.register_config(
    name='default',
    components={
        'cloud': (_RADIAL_CUTOFF, 250),
        'band1': (_RADIAL_CUTOFF, 50),
        'band2': (_RADIAL_CUTOFF, 50),
        'band3': (_RADIAL_CUTOFF, 50),
        'ring': (2.25, 50),
        'feature': (1, 50),
    }
)
integration_configs.register_config(
    name='high',
    components={
        'cloud': (_RADIAL_CUTOFF, 500),
        'band1': (_RADIAL_CUTOFF, 500),
        'band2': (_RADIAL_CUTOFF, 500),
        'band3': (_RADIAL_CUTOFF, 500),
        'ring': (2.25, 200),
        'feature': (1, 200),
    }
)
integration_configs.register_config(
    name='fast',
    components={
        'cloud': (_RADIAL_CUTOFF, 25),
        'band1': (_RADIAL_CUTOFF, 25),
        'band2': (_RADIAL_CUTOFF, 25),
        'band3': (_RADIAL_CUTOFF, 25),
        'ring': (2.25, 25),
        'feature': (1, 25),
    }
)