from collections.abc import Iterable
from typing import Optional, Union, Iterable
from datetime import datetime

import astropy.units as u
import numpy as np

from zodipy import models
from zodipy import _coordinates as coords
from zodipy import _integration as integ
from zodipy import simulation 

class Zodi:
    """Interface for simulating the Zodiacal emission.
    
    Currently, Zodipy only supports simulation the instantaneous Zodiacal 
    emission. It is possible that TOD simulations will be implemented in 
    future.
    """

    def __init__(
        self, 
        observer : Optional[str] = 'L2',
        observation_times : Optional[Union[Iterable[datetime], datetime]] = None,
        model : Optional[models.InterplanetaryDustModel] = models.PLANCK_2018,
        integration_config : Optional[integ.IntegrationConfig] = integ.DEFAULT,
    ) -> None:
        """Initializing the zodi interface.

        The geometric setup of the simulation, the IPD model, and the 
        integration configuration used when integrating up the emission 
        are all configured here in the initialization of the Zodi object.
        
        Parameters
        ----------
        observer : str, optional
            The observer. Defaults to L2.
        observation_times : Iterable, optional
            The times of observation. Must be an iterable containing 
            `datetime` objects. Defaults to a single observeration at the 
            current time.
        model : `zodipy.models.Model`, optional
            The Interplanteary dust model used in the simulation. Defaults 
            to the model used in the Planck 2018 analysis.
        integ : `zodipy.integration.IntegrationConfig`, optional
            Integration config object determining the integration details
            used in the simulation.
        """

        if observation_times is None:
            observation_times = [datetime.now().date]
        elif not isinstance(observation_times, Iterable): 
            observation_times = [observation_times]

        observer_locations = [
            coords.get_target_coordinates(observer, time) 
            for time in observation_times
        ]
        earth_locations = [
            coords.get_target_coordinates('earth', time) 
            for time in observation_times
        ]

        self.simulation_strategy = simulation.InstantaneousStrategy(
            model, integration_config, observer_locations, earth_locations
        )

    def get_emission(
        self, 
        nside: int, 
        freq: Union[float, u.Quantity], 
        coord: Optional[str] = 'G',
        return_comps: Optional[bool] = False
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

        Returns
        -------
        emission : `numpy.ndarray`
            Simulated Zodiacal emission in units of MJy/sr.
        """

        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        emission = self.simulation_strategy.simulate(nside, freq)

        if coord != 'E':
            emission = coords.change_coordinate_system(emission, coord)

        return emission if return_comps else emission.sum(axis=0)