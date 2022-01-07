from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from zodipy._components import Component
from zodipy._labels import Label, LABEL_TO_CLASS
from zodipy._source_parameters import SourceParameter


class IPDModel:
    """An Interplanetary Dust Model.

    An IPDModel is a container for a unique combination of initialized Zodiacal
    Components and SourceParameters fits."""

    def __init__(
        self,
        name: str,
        component_parameters: Dict[Label, Dict[str, float]],
        interplanetary_temperature: float,
        delta: float,
        emissivity: Optional[SourceParameter] = None,
        albedo: Optional[SourceParameter] = None,
    ) -> None:
        """Initializes an IPDModel given a set of parameters and emissivities."""

        self.name = name
        self.components: Dict[Label, Component] = {}
        for label, parameters in component_parameters.items():
            component_class = LABEL_TO_CLASS[label]
            self.components[label] = component_class(**parameters)

        self.interplanetary_temperature = interplanetary_temperature
        self.delta = delta
        self.emissivity = emissivity
        self.albedo = albedo

    @property
    def includes_earth_neighboring_components(self) -> bool:
        """Returns True if the model includes an Earth-neighboring component."""

        return Label.RING in self.components or Label.FEATURE in self.components

    def __getitem__(self, component_name: str) -> Component:
        """Returns a sky component from the cosmoglobe model."""

        return self.components[Label(component_name)]


@dataclass
class IPDModelRegistry:
    """Container for registered IPDModels."""

    registry: Dict[str, IPDModel] = field(default_factory=dict)

    def register_model(self, model: IPDModel) -> None:
        """Adds a new IPD model to the registry."""

        if (name := model.name) in self.registry:
            raise ValueError(f"model by name {name} is already registered.")

        self.registry[name] = model

    def get_model(self, name: str) -> IPDModel:
        """Returns a IPDModel from the registry."""

        if name not in self.registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self.registry)}"
            )

        return self.registry[name]


model_registry = IPDModelRegistry()
