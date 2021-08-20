from typing import Optional, Union, Iterable, Dict

import astropy.units as u
import numpy as np

from zodipy._coordinates import get_target_coordinates, change_coordinate_system
from zodipy._integration import INTEGRATION_CONFIGS
from zodipy.models import MODELS
from zodipy.simulation import InstantaneousStrategy, TimeOrderedStrategy


class Zodi:
    """The Zodipy interface."""

    def __init__(
        self, 
        observer: Optional[str] = 'L2',
        epochs: Optional[Union[float, Iterable[float], Dict[str, str]]] = None,
        hit_maps: Optional[Iterable[np.ndarray]] = None,
        model: Optional[str] = 'planck 2018',
        integration_config: Optional[str] = 'default'
    ) -> None:
        """Setting up initial conditions and other simulation specifics.

        Parameters
        ----------
        observer
            The observer. Defaults to 'L2'.
        epochs
            Either a list of epochs in JD or MJD format or a dictionary
            defining a range of times and dates; the range dictionary has to
            be of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
            ``'stop'``:'YYYY-MM-DD [HH:MM:SS]', ``'step'``:'n[y|d|h|m|s]'}.
            Epoch timescales depend on the type of query performed: UTC for
            ephemerides queries, TDB for element queries, CT for vector queries.
            If no epochs are provided, the current time is used.
        hit_maps
            The number of times each pixel is observed for a given observation.
        model
            String referencing the Interplanetary dust model used in the 
            simulation. Available options are 'planck 2013', 'planck 2015',
            and 'planck 2018'. Defaults to 'planck 2018'.
        integration_config
            String referencing the integration_config object used when 
            calling `get_emission`. Available options are: 'default', and 
            'high'. Defaults to 'default'.
        """

        try:
            model = MODELS[model.lower()]
        except KeyError:
            raise KeyError(
                f"Model {model!r} not found. Available models are: "
                f"{list(MODELS.keys())}"
        )
        try:
            integration_config = INTEGRATION_CONFIGS[integration_config.lower()]
        except KeyError:
            raise KeyError(
                f"Config {integration_config!r} not found. Available configs "
                f"are: {list(INTEGRATION_CONFIGS.keys())}"
        )
        
        observer_locations = get_target_coordinates(observer, epochs) 
        earth_locations = get_target_coordinates('earth', epochs) 
        
        # We check that the dimensons of hit_map corresponds to the number
        # of observations.
        if hit_maps is not None:
            if np.ndim(hit_maps) == 1:
                hit_maps = np.expand_dims(hit_maps, axis=0)
            elif len(hit_maps) != len(observer_locations):
                raise ValueError(
                    f"Lenght of 'hit_maps' ({len(hit_maps)}) are not matching "
                    f"the number of observations ({len(observer_locations)})"
                )

        if len(observer_locations) == 1:
            self._simulation_strategy = InstantaneousStrategy(
                model, 
                integration_config, 
                observer_locations[0], 
                earth_locations[0], 
                hit_maps
            )
        else:
            self._simulation_strategy = TimeOrderedStrategy(
                model, 
                integration_config, 
                observer_locations, 
                earth_locations, 
                hit_maps
            )     

    def get_emission(
        self, 
        nside: int, 
        freq: Union[float, u.Quantity], 
        coord: Optional[str] = 'G',
        return_comps: Optional[bool] = False,
    ) -> np.ndarray:
        """Simulates the Zodiacal emission in units of MJy/sr.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq 
            Frequency at which to evaluate the Zodiacal emission. The 
            frequency should be in units of GHz unless an 
            `astropy.units.Quantity` object is passed for which it only 
            needs to be compatible with Hz.
        coord
            Coordinate system of the output map. Available options are: 
            'E', 'C', or 'G'. Defaults to 'G'.
        return_comps
            If True, the emission of each component in the model is 
            returned separatly in the first dim of the output array. 
            Defaults to False.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of MJy/sr.
        """

        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        emission = self._simulation_strategy.simulate(
            nside, freq
        )
        emission = change_coordinate_system(emission, coord)

        return emission if return_comps else emission.sum(axis=0)