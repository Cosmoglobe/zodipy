from dataclasses import dataclass, field
from typing import Any, Dict, List

from zodipy._components import Component
from zodipy._labels import Label, LABEL_TO_CLASS


@dataclass
class InterplanetaryDustModel:
    name: str
    comp_labels: List[Label]
    comp_params: Dict[Label, Dict[str, Any]]
    spectral_params: Dict[Any, Any]
    source_params: Dict[str, Any]
    doc: str = ""
    comps: Dict[Label, Component] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize all components."""

        if Label.CLOUD not in self.comp_labels:
            raise ValueError(
                "The cloud component is required component for the K98 model"
            )

        for label in self.comp_labels:
            parameters = self.comp_params[label]
            component_class = LABEL_TO_CLASS[label]
            self.comps[label] = component_class(**parameters)

    @property
    def includes_ring(self) -> bool:
        """Returns True if the model includes an Earth-neighboring component."""

        return Label.RING in self.comp_labels or Label.FEATURE in self.comp_labels

    @property
    def ncomps(self) -> int:
        """Returns the number of components in the model."""

        return len(self.comp_labels)

    def __str__(self) -> str:
        """String representation of the Interplanetary dust model."""

        reprs = []
        for label in self.comp_labels:
            reprs.append(f"{label.value.capitalize()}\n")

        main_repr = "InterplanetaryDustModel("
        main_repr += f"\n  name: {self.name}"
        main_repr += f"\n  info: {self.doc}"
        main_repr += "\n  components: "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += ")"

        return main_repr


@dataclass
class ModelRegistry:
    """Container for registered InterplanetaryDustModel's."""

    _registry: Dict[str, InterplanetaryDustModel] = field(default_factory=dict)

    def register_model(
        self,
        name: str,
        comp_labels: List[Label],
        comp_params: Dict[Label, Dict[str, Any]],
        spectral_params: Dict[Any, Any],
        source_params: Dict[str, Any],
        doc: str = "",
    ) -> None:
        if name in self._registry:
            raise ValueError(f"a model by the name {name!s} is already registered.")

        self._registry[name] = InterplanetaryDustModel(
            name,
            comp_labels,
            comp_params,
            spectral_params,
            source_params,
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
