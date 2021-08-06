from dataclasses import dataclass
from typing import Dict

from zodipy import CompLabel
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
        CompLabel.CLOUD : components.Cloud(parameters=K98[CompLabel.CLOUD]),
        CompLabel.BAND1 : components.Band(parameters=K98[CompLabel.BAND1]),
        CompLabel.BAND2 : components.Band(parameters=K98[CompLabel.BAND2]),
        CompLabel.BAND3 : components.Band(parameters=K98[CompLabel.BAND3]),
        CompLabel.RING : components.Ring(parameters=K98[CompLabel.RING]),
        CompLabel.FEATURE : components.Feature(parameters=K98[CompLabel.FEATURE]),
    },
    emissivities=emissivities.PLANCK_2013,
)

PLANCK_2015 = Model(
    components={
        CompLabel.CLOUD : components.Cloud(parameters=K98[CompLabel.CLOUD]),
        CompLabel.BAND1 : components.Band(parameters=K98[CompLabel.BAND1]),
        CompLabel.BAND2 : components.Band(parameters=K98[CompLabel.BAND2]),
        CompLabel.BAND3 : components.Band(parameters=K98[CompLabel.BAND3]),
    },
    emissivities=emissivities.PLANCK_2015,
)

PLANCK_2018 = Model(
    components={
        CompLabel.CLOUD : components.Cloud(parameters=K98[CompLabel.CLOUD]),
        CompLabel.BAND1 : components.Band(parameters=K98[CompLabel.BAND1]),
        CompLabel.BAND2 : components.Band(parameters=K98[CompLabel.BAND2]),
        CompLabel.BAND3 : components.Band(parameters=K98[CompLabel.BAND3]),
    },
    emissivities=emissivities.PLANCK_2018,
)