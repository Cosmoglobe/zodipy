from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zodipy._components import Component, ComponentLabel
from zodipy._emissivities import Emissivities


@dataclass
class InterplanetaryDustModel:
    """Class representing a initialized IPD Model."""

    components: Dict[ComponentLabel, Component]
    emissivities: Optional[Emissivities] = None

    def __getitem__(self, component: str) -> Component:
        """Returns a sky component from the cosmoglobe model."""

        return self.components[ComponentLabel(component)]


@dataclass
class ModelRegistry:
    """Container for registered IPD models."""

    REGISTRY: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        components: List[ComponentLabel],
        parameters: Dict[ComponentLabel, Dict[str, float]],
        emissivities: Optional[Emissivities] = None,
    ) -> None:
        """Adds a new IPD model to the registry."""

        if name in self.REGISTRY:
            raise ValueError(f"model by name {name} is already registered.")

        initialized_components: Dict[ComponentLabel, Component] = {}
        for component in components:
            initialized_components[component] = component.value(**parameters[component])

        self.REGISTRY[name] = InterplanetaryDustModel(
            components=initialized_components, emissivities=emissivities
        )

    def get_model(self, name: str) -> InterplanetaryDustModel:
        """Returns a IPD model from the registry."""

        return self.REGISTRY[name]
