from dataclasses import dataclass
from typing import Dict

from zodipy import components
from zodipy.data import parameters, emissivities


class ModelError(Exception):
    """Error raised in the case there is a Model related exception."""


@dataclass
class Model:
    components : Dict[str, components.BaseComponent]
    emissivities : dict

    def __post_init__(self) -> None:
        for component in self.components.values():
            if not issubclass(component.__class__, components.BaseComponent):
                raise ModelError(
                    f'Component {component} is not a subclass of ' 
                    f'{components.BaseComponent}'
                )


K98 = parameters.K98

PLANCK_2013 = Model(
    components={
        'cloud' : components.Cloud(K98['cloud']),
        'band1' : components.Band(K98['band1']),
        'band2' : components.Band(K98['band2']),
        'band3' : components.Band(K98['band3']),
        'ring' : components.Ring(K98['ring']),
        'feature' : components.Feature(K98['feature']),
    },
    emissivities=emissivities.PLANCK_2013,
)

PLANCK_2015 = Model(
    components={
        'cloud' : components.Cloud(K98['cloud']),
        'band1' : components.Band(K98['band1']),
        'band2' : components.Band(K98['band2']),
        'band3' : components.Band(K98['band3']),
    },
    emissivities=emissivities.PLANCK_2015,
)

PLANCK_2018 = Model(
    components={
        'cloud' : components.Cloud(K98['cloud']),
        'band1' : components.Band(K98['band1']),
        'band2' : components.Band(K98['band2']),
        'band3' : components.Band(K98['band3']),
    },
    emissivities=emissivities.PLANCK_2018,
)