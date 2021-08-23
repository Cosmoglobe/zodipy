from typing import Optional, Union, Iterable, Dict

import astropy.units as u
import numpy as np

from zodipy._coordinates import get_target_coordinates, to_frame
from zodipy._simulation import InstantaneousStrategy, TimeOrderedStrategy
from zodipy.integration_configs import integration_configs
from zodipy.models import models


class Zodi:
    """The Zodipy interface.

    The Zodiacal emission seen by an observer is highly dependant on the 
    specifics of the observation. As an observer moves through the Solar 
    System, they will look through different columns of interplanetary 
    dust. 
    
    In Zodipy, the simulated Zodiacal emission is a pixel-weighted average 
    of all included observations. For each observation, a line-of-sight 
    integral is carried out for all observed pixels from the observer 
    location. If no hit_counts are provided, the full sky average over all 
    observations are returned.
    
    Parameters
    ----------
    observer
        The observer. Defaults to 'L2'.
    epochs
        The observeration times given as a single epoch, or a list of epochs 
        in JD or MJD format, or a dictionary defining a range of times and
        dates; the range dictionary has to be of the form 
        {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]', 
        'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time 
        is used in UTC.
    hit_counts
        The number of times each pixel is hit during each observation
    model
        The Interplanetary dust model used in the simulation. Available 
        options are 'planck 2013', 'planck 2015', and 'planck 2018'. 
        Defaults to 'planck 2018'.
    integration_config
        The integration_config object used when calling `get_emission`. 
        Available options are: 'default', and 'high', and 'fast'. Defaults 
        to 'default'.
    """

    def __init__(
        self, 
        observer: Optional[str] = 'L2',
        epochs: Optional[Union[float, Iterable[float], Dict[str, str]]] = None,
        hit_counts: Optional[Iterable[np.ndarray]] = None,
        model: Optional[str] = 'planck 2018',
        integration_config: Optional[str] = 'default'
    ) -> None:

        model = models.get_model(model)
        integration_config = integration_configs.get_config(integration_config)

        observer_locations = get_target_coordinates(observer, epochs) 
        earth_locations = get_target_coordinates('earth', epochs) 
        
        number_of_observations = len(observer_locations)
        if hit_counts is not None:
            hit_counts = np.asarray(hit_counts)
            number_of_hit_counts = 1 if np.ndim(hit_counts) == 1 else len(hit_counts)
            if number_of_hit_counts != number_of_observations:
                raise ValueError(
                    f"The number of 'hit_counts' ({number_of_hit_counts}) are not "
                    "matching the number of observations "
                    f"({number_of_observations})"
                )

        if number_of_observations == 1:
            simulation_strategy = InstantaneousStrategy
            observer_locations = observer_locations.squeeze()
            earth_locations = earth_locations.squeeze()
        else:
            simulation_strategy = TimeOrderedStrategy
        self._simulation_strategy = simulation_strategy(
            model, 
            integration_config, 
            observer_locations, 
            earth_locations, 
            hit_counts
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
            Coordinate frame of the output map. Available options are: 
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

        emission = self._simulation_strategy.simulate(nside, freq)
        emission = to_frame(emission, coord)

        return emission if return_comps else emission.sum(axis=0)