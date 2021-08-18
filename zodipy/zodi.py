from typing import Optional, Union
from datetime import datetime

import astropy.units as u
import numpy as np

from zodipy._coordinates import get_target_coordinates, change_coordinate_system
from zodipy._integration import INTEGRATION_CONFIGS
from zodipy.models import MODELS
from zodipy.simulation import InstantaneousStrategy


class Zodi:
    """The main Zodipy interface.
    
    Initializing this class sets up the initial conditions for the
    simulation problem. The `get_emission` method is called to perfrom 
    the simulation.
    """

    def __init__(
        self, 
        observer: Optional[str] = 'L2',
        start: Optional[datetime] = datetime.now().date(),
        stop: Optional[datetime] = None,
        step: Optional[str] = '1d',
        model: Optional[str] = 'planck 2018',
        integration_config: Optional[str] = 'default'
    ) -> None:
        """Initializing the zodi interface.

        Parameters
        ----------
        observer
            The observer. Defaults to 'L2'.
        start
            Datetime object representing the time of observation. Defaults
            to the current date.
        stop
            Datetime object represetning the time when the observation ended.
            If None, a single observation is assumed. Defaults to None.
        step
            Step size between the start and stop dates in either days 
            denoted by 'd' or hours 'h'. Defaults to '1d'.
        model
            String referencing the Interplanetary dust model used in the 
            simulation. Available options are 'planck 2013', 'planck 2015',
            and 'planck 2018'. Defaults to 'planck 2018'.
        integration_config
            String referencing the integration_config object used when 
            calling `get_emission`. Available options are: 'default', and 
            'high'. Defaults to 'default'.
        """

        observer_locations = get_target_coordinates(
            observer, start, stop, step
        ) 
        earth_locations = get_target_coordinates(
            'earth', start, stop, step
        ) 

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

        self.simulation_strategy = InstantaneousStrategy(
            model, integration_config, observer_locations, earth_locations
        )

    def get_emission(
        self, 
        nside: int, 
        freq: Union[float, u.Quantity], 
        coord: Optional[str] = 'G',
        return_comps: Optional[bool] = False,
        solar_cut: float = None
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
        solar_cut
            Angle in degrees between observer and the Sun for which all 
            pixels are masked for each observation.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of MJy/sr.
        """

        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        emission = self.simulation_strategy.simulate(nside, freq, solar_cut)

        emission = change_coordinate_system(emission, coord)

        return emission if return_comps else emission.sum(axis=0)