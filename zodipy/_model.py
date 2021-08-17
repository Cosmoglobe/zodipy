from typing import Iterable, Dict

from zodipy._emissivity import Emissivity
from zodipy import components


class InterplanetaryDustModel:
    """Class that represents a model of the IPD.
    
    Attributes
    ----------
    components : dict
        Dictionary containing initialized `zodipy.components.BaseComponent`
        objects.
    emissivities : `zodipy._emissivity.Emissivity`
        Emissivity object.
    """

    def __init__(
        self,
        components: Iterable[str], 
        parameters: Dict[str, Dict[str, float]], 
        emissivities: Emissivity
    ) -> None: 
        """Initilizes a Model object.
        
        Parameters
        ----------
        components
            Iterable containing component labels as strings.
        parameters
            Dictionary containing component parameters.
        emissivities
            Emissivity object.
        """
        
        self.components = self._init_components(components, parameters)
        self.emissivities = emissivities

    def _init_components(self, comp_labels: Iterable, parameters: dict) -> dict:
        """Initialize component dictionary."""

        comps = {}
        for label in comp_labels:
            if label.startswith('cloud'):
                comp_type = components.Cloud
            elif label.startswith('band'):
                comp_type = components.Band
            elif label.startswith('ring'):
                comp_type = components.Ring
            elif label.startswith('feature'):
                comp_type = components.Feature

            comps[label] = comp_type(**parameters[label])

        return comps