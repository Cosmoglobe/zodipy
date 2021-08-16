from typing import Optional, Union
from datetime import datetime

import astropy.units as u
import numpy as np

from zodipy import models
from zodipy import simulation 
from zodipy import _coordinates
from zodipy import _integration

class Zodi:
    """Interface for simulating the Zodiacal emission.
    
    Currently, Zodipy only supports simulation the instantaneous Zodiacal 
    emission. It is possible that TOD simulations will be implemented in 
    future.
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

        The geometric setup of the simulation, the IPD model, and the 
        integration configuration used when integrating up the emission 
        are all configured here in the initialization of the Zodi object.
        
        Parameters
        ----------
        observer : str, optional
            The observer. Defaults to L2.
        start : `datetime.datetime`, optional
            Datetime object representing the time of observation. Defaults
            to the current date.
        stop : `datetime.datetime`, optional
            Datetime object represetning the time when the observation ended.
            If None, the observation is assumed to only occur on the start 
            date. Defaults to None.
        step : str
            Step size from the start to stop dates in days denoted by 'd' 
            or hours 'h'. Defaults to 1 day ('1d').
        model : str, optional
            String representing the Interplanteary dust model used in the 
            simulation. Available options are 'planck 2013', 'planck 2015',
            and 'planck 2018'. Defaults to 'planck 2018'.
        integration_config : str, optional
            String representing the integration config which determins the 
            integration details used in the simulation. Available options
            'default', and 'high'. Defaults to 'default'.
        """

        observer_locations = _coordinates.get_target_coordinates(
            observer, start, stop, step
        ) 
        earth_locations = _coordinates.get_target_coordinates(
            'earth', start, stop, step
        ) 

        if 'planck' in model.lower():
            if '2013' in model:
                model = models.PLANCK_2013
            elif '2015' in model:
                model = models.PLANCK_2015
            elif '2018' in model:
                model = models.PLANCK_2018
            else:
                raise ValueError(
                    "Available models are: 'planck 2013', 'planck 2015', and "
                    "'planck 2018'"
                )
        
        if integration_config == 'default':
            integration_config = _integration.DEFAULT
        elif integration_config == 'high':
            integration_config = _integration.HIGH
        else:
            raise ValueError(
                "Available configs are: 'default' and 'high'"
            )

        self.simulation_strategy = simulation.InstantaneousStrategy(
            model, integration_config, observer_locations, earth_locations
        )

    def get_emission(
        self, 
        nside: int, 
        freq: Union[float, u.Quantity], 
        coord: Optional[str] = 'G',
        return_comps: Optional[bool] = False,
        mask: float = None
    ) -> np.ndarray:
        """Returns the simulated Zodiacal emission in units of MJy/sr.

        Parameters
        ----------
        nside : int
            HEALPIX map resolution parameter.
        freq : float, `astropy.units.Quantity`
            Frequency [GHz] at which to evaluate the IPD model. The 
            frequency should be in units of GHz, unless an `astropy.Quantity`
            object is used, for which it only needs to be compatible with Hz.
        coord : str, optional
            Coordinate system of the output map. Accepted inputs are: 'E', 
            'C', or 'G'. Defaults to 'G' which is the Galactic coordinate 
            system.
        return_comps : bool, optional
            If True, the emission of each component in the model is returned
            separatly in form of an array of shape (`n_comps`, `npix`). 
            Defaults to False.
        mask : float, optional
            Angle [deg] between observer and the Sun for which all pixels 
            are masked at each observation. A mask of 90 degrees can be 
            selected to simulate an observer that never looks inwards the Sun.

        Returns
        -------
        emission : `numpy.ndarray`
            Simulated Zodiacal emission in units of MJy/sr.
        """

        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        emission = self.simulation_strategy.simulate(nside, freq, mask)

        emission = _coordinates.change_coordinate_system(emission, coord)

        return emission if return_comps else emission.sum(axis=0)