from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from zodipy._components import Component
from zodipy._component_labels import ComponentLabel, LABEL_TO_CLASS
from zodipy._emissivities import Emissivity
from zodipy._exceptions import ComponentNotImplemented


class IPDModel:
    """An IPD Model, composed of a combination of Zodiacal Components and optionally emissivities."""

    def __init__(
        self,
        name: str,
        parameters: Dict[ComponentLabel, Dict[str, float]],
        emissivities: Optional[Emissivity] = None,
    ) -> None:
        """Initializes an IPDModel given a set of parameters and optionally emissivities."""

        self.name = name
        self.components: Dict[ComponentLabel, Component] = {}
        for component_label, component_parameters in parameters.items():
            component_class = LABEL_TO_CLASS[component_label]
            self.components[component_label] = component_class(**component_parameters)

        self.emissivities = emissivities

    @property
    def includes_earth_neighboring_components(self) -> bool:
        """Returns True if the model includes a earth neighboring component."""

        return (
            ComponentLabel.RING in self.components
            or ComponentLabel.FEATURE in self.components
        )

    def __getitem__(self, component_name: str) -> Component:
        """Returns a sky component from the cosmoglobe model."""

        return self.components[ComponentLabel(component_name)]

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


def register_custom_model(
    name: str,
    parameters: Dict[str, Dict[str, float]],
    emissivities: Optional[Dict[str, Sequence[float]]] = None,
    emissivities_freq: Optional[Sequence[float]] = None,
) -> None:
    """Registers a custom model to the registry of Interplanetary Dust Models.
    
    Parameters
    ----------
    name
        Name representing the custom model.
    parameters
        Dictionary containing parameter key, value pairs for each component in
        the model. 
    emissivities
        Dictionary containing the emissivity fits for each component in the 
        model.
    emissivities_freq
        Frequencies corresponding to the emissivity fits.
    """

    _parameters: Dict[ComponentLabel, Dict[str, float]] = {}
    for component_label, component_parameters in parameters.items():
        if not isinstance(component_label, ComponentLabel):
            try:
                component_label = ComponentLabel(component_label)
            except ValueError:
                raise ComponentNotImplemented(
                    "component not found. Available components are: "
                    f"{', '.join(ComponentLabel.__members__.keys())}"
                )

        _parameters[component_label] = component_parameters

    if not emissivities:
        emissivity = None

    else:
        if emissivities_freq is None:
            raise ValueError("frequencies must be specified for the emissivities.")

        _emissivities: Dict[ComponentLabel, Sequence[float]] = {}
        for component_label, component_emissivities in emissivities.items():
            if not isinstance(component_label, ComponentLabel):
                try:
                    component_label = ComponentLabel(component_label)
                except ValueError:
                    raise ComponentNotImplemented(
                        "component not found. Available components are: "
                        f"{', '.join(ComponentLabel.__members__.keys())}"
                    )
            _emissivities[component_label] = component_emissivities

        emissivity = Emissivity(frequencies=emissivities_freq, components=_emissivities)
    model = IPDModel(name=name, parameters=_parameters, emissivities=emissivity)

    model_registry.register_model(model)

    