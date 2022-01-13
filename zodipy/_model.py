from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from zodipy._labels import Label, LABEL_TO_CLASS
from zodipy._components import Component


@dataclass
class InterplanetaryDustModel:
    name: str
    components: List[Label]
    component_parameters: Dict[Label, Dict[str, float]]
    source_component_parameters: Dict[str, Dict[Union[str, Label], Any]]
    source_parameters: Dict[str, Any]
    doc: str = ""

    def get_initialized_component(self, label: Label) -> Component:
        """Initializes and returns a Zodiacal Component from the model parameters."""

        if label not in self.components:
            raise ValueError(f"{label.value} is not included in the {self.name} model")

        parameters = self.component_parameters[label]
        component_class = LABEL_TO_CLASS[label]

        return component_class(**parameters)

    @property
    def includes_ring(self) -> bool:
        """Returns True if the model includes an Earth-neighboring component."""

        return Label.RING in self.components or Label.FEATURE in self.components


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        components: List[Label],
        component_parameters: Dict[Label, Dict[str, float]],
        source_component_parameters: Dict[str, Dict[Union[str, Label], Any]],
        source_parameters: Dict[str, Any],
        doc: str = "",
    ) -> None:
        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = InterplanetaryDustModel(
            name,
            components,
            component_parameters,
            source_component_parameters,
            source_parameters,
            doc,
        )

    def get_model(self, name: str) -> InterplanetaryDustModel:
        if name not in self._registry:
            raise ModuleNotFoundError(
                f"{name} is not a model in the registry. Avaliable models are "
                f"{', '.join(self._registry)}"
            )

        return self._registry[name]


model_registry = ModelRegistry()