from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from zodipy._components import Component
from zodipy._component_labels import ComponentLabel, LABEL_TO_CLASS
from zodipy._emissivities import Emissivity


@dataclass
class InterplanetaryDustModel:
    """Class representing a initialized IPD Model."""

    components: Dict[ComponentLabel, Component]
    emissivities: Optional[Emissivity] = None

    def __getitem__(self, component_name: str) -> Component:
        """Returns a sky component from the cosmoglobe model."""

        return self.components[ComponentLabel(component_name)]


@dataclass
class InterplanetaryDustModelRegistry:
    """Container for registered IPD models."""

    registry: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        components: List[ComponentLabel],
        parameters: Dict[ComponentLabel, Dict[str, float]],
        emissivities: Optional[Emissivity] = None,
    ) -> None:
        """Adds a new IPD model to the registry."""

        if name in self.registry:
            raise ValueError(f"model by name {name} is already registered.")

        initialized_components: Dict[ComponentLabel, Component] = {}
        for component_label in components:
            component_class = LABEL_TO_CLASS[component_label]
            initialized_components[component_label] = component_class(
                **parameters[component_label]
            )

        self.registry[name] = InterplanetaryDustModel(
            components=initialized_components, emissivities=emissivities
        )

    def get_model(self, name: str) -> InterplanetaryDustModel:
        """Returns a IPD model from the registry."""

        if name not in self.registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self.registry)}"
            )
            
        return self.registry[name]
