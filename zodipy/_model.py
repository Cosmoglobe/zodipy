from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from zodipy._labels import Label, LABEL_TO_CLASS
from zodipy._components import Component


@dataclass
class InterplanetaryDustModel:
    name: str
    component_labels: List[Label]
    component_parameters: Dict[Label, Dict[str, Any]]
    source_component_parameters: Dict[str, Dict[Union[str, Label], Any]]
    source_parameters: Dict[str, Any]
    doc: str = ""
    components: Dict[Label, Component] = field(default_factory=dict)


    def __post_init__(self) -> None:
        """Initialize all components."""
        
        for label in self.component_labels:
            parameters = self.component_parameters[label]
            component_class = LABEL_TO_CLASS[label]
            self.components[label] = component_class(**parameters)

    @property
    def includes_ring(self) -> bool:
        """Returns True if the model includes an Earth-neighboring component."""

        return Label.RING in self.component_labels or Label.FEATURE in self.component_labels

    @property
    def ncomps(self) -> int:
        """Returns the number of components in the model."""

        return len(self.component_labels)

@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        component_labels: List[Label],
        component_parameters: Dict[Label, Dict[str, Any]],
        source_component_parameters: Dict[str, Dict[Union[str, Label], Any]],
        source_parameters: Dict[str, Any],
        doc: str = "",
    ) -> None:
        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = InterplanetaryDustModel(
            name,
            component_labels,
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

    def get_registered_model_names(self) -> List[str]:
        return list(self._registry.keys())


model_registry = ModelRegistry()