from typing import Iterable

from zodipy._emissivity import Emissivity
from zodipy import components


class Model:
    
    def __init__(
        self,
        components: Iterable[str], 
        parameters: dict, 
        emissivities: Emissivity
    ) -> None: 
        """Class that represents a model of the IPD.
        
        Parameters
        ----------
        components : Iterable
            Iterable containing component labels as strings.
        parameters : dict
            Dictionary containing component parameters.
        emissivities : `zodipy.emissivities.Emissivity`
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