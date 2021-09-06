from dataclasses import dataclass
from typing import Iterable, Dict, Union, Tuple

from zodipy._components import BaseComponent, Cloud, Band, Ring, Feature
from zodipy._emissivities import Emissivities

ModelParameterType = Dict[str, Dict[str, float]]
EmissivitiesType = Dict[str, Union[Tuple[float], Dict[str, Tuple[float]]]]


@dataclass
class Model:
    """Data class representing an interplanetary dust model."""

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
        parameters: ModelParameterType,
        emissivities: EmissivitiesType,
    ) -> None:
        """Initializes and stores a model."""

        model = _init_model(components, parameters, emissivities)
        self._models[name] = model

    def get_model(self, name: str) -> Model:
        """Returns a registered model."""

        model = self._models.get(name)
        if model is None:
            raise ValueError(
                f"model {name} is not registered. Available models are "
                f"{list(self._models.keys())}"
            )

        return model


def _init_model(components, parameters, emissivities):
    """Handles mapping of compoents to correct classes."""
    
    initialized_components = {}
    for comp in components:
        if comp.startswith("cloud"):
            comp_type = Cloud
        elif comp.startswith("band"):
            comp_type = Band
        elif comp.startswith("ring"):
            comp_type = Ring
        elif comp.startswith("feature"):
            comp_type = Feature
        initialized_components[comp] = comp_type(**parameters[comp])

    emissivities = Emissivities(**emissivities)

    return Model(
        components=initialized_components, emissivities=emissivities
    )
