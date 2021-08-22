from dataclasses import dataclass
from typing import Iterable, Dict, Union, Tuple

from zodipy._emissivity import Emissivities
from zodipy.components import BaseComponent, Cloud, Band, Ring, Feature


@dataclass
class InterplanetaryDustModel:
    """Data class representing an Interplanetary dust model."""

    components: Dict[str, BaseComponent]
    emissivities: Emissivities


class ModelFactory:
    """Factory responsible for registring and book-keeping models."""

    def __init__(self) -> None:
        self._models = {}

    def register_model(
        self, 
        name: str, 
        components: Iterable[str], 
        parameters: Dict[str, Dict[str, float]],
        emissivities: Dict[str, Union[Tuple[float], Dict[str, Tuple[float]]]]
    ) -> None:
        """Initializes and stores an IPD model."""

        model = init_model(components, parameters, emissivities)
        self._models[name] = model

    def get_model(self, name: str) -> InterplanetaryDustModel: 
        """Returns a registered model."""
        
        model = self._models.get(name)
        if model is None:
            raise ValueError(
                f'model {name} is not registered. Available models are '
                f'{list(self._models.keys())}'
            )

        return model


def init_model(components, parameters, emissivities):
    initialized_components = {}
    for comp in components:
        if comp.startswith('cloud'):
            comp_type = Cloud
        elif comp.startswith('band'):
            comp_type = Band
        elif comp.startswith('ring'):
            comp_type = Ring
        elif comp.startswith('feature'):
            comp_type = Feature
        initialized_components[comp] = comp_type(**parameters[comp])
        
    emissivities = Emissivities(**emissivities)

    return InterplanetaryDustModel(
        components=initialized_components,
        emissivities=emissivities
    )