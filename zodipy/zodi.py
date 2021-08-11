from typing import Optional, Union
from datetime import datetime

import astropy.units as u
import healpy as hp
import numpy as np

from zodipy import models
from zodipy import _coordinates
from zodipy import _integration as integ


class Zodi:
    """Interface for simulating Zodiacal emission."""

    def __init__(
        self, 
        observer : Optional[str] = 'L2',
        observation_time : Optional[datetime] = datetime.now().date(),
        earth_position : Optional[np.ndarray] = None,
        model : Optional[models.Model] = models.PLANCK_2018,
        integ : Optional[integ.IntegrationConfig] = integ.DEFAULT_CONFIG,
    ) -> None:
        """Initializing the zodi interface.
        
        Parameters
        ----------
        observer : str, optional
            The observer. Defaults to L2.
        observation_time : `datetime.datetime`, optional
            The time of the observation. Defaults to the current time.
        earth_position : `numpy.ndarray`, optional
            Heliocentric coordinates of the Earth. If None, Earth's 
            coordinates from the observation_time is used. Defaults to None.
        model : `zodipy.models.Model`, optional
            The Interplanteary dust model used in the simulation. 
            Defaults to the model used in the Planck 2018 analysis.
        integ : `zodipy.integration.IntegrationConfig`, optional
            Integration config object determining the integration details
            used in the simulation. Defaults to 
            `zodipy._integration.DEFAULT_CONFIG`.
        """

        self.X_observer = _coordinates.get_target_coordinates(
            observer, observation_time
        )
        if earth_position is not None:
            self.X_earth = earth_position
        else:
            self.X_earth = _coordinates.get_target_coordinates(
                'earth', observation_time
            )

        self.model = model
        self.integ = integ

    def simulate(
        self, nside: int, freq: Union[float, u.Quantity], coord: str = 'G'
    ) -> np.ndarray:
        """Simulates the Zodiacal emission given a frequency.

        Parameters
        ----------
        nside : int
            HEALPIX map resolution parameter.
        freq : float, `astropy.units.Quantity`
            Frequency [GHz] at which to evaluate the IPD model. The 
            frequency should be in units of GHz, unless an `astropy.Quantity`
            object is used, for which it only needs to be compatible with Hz.
        coord : str, optional
            Coordinate system of the output map. Defaults to 'G' which is 
            the Galactic coordinate system.

        Returns
        -------
        emission : `numpy.ndarray`
            Simulated Zodiacal emission for some nside at some frequency.
        """
        
        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        X_observer = self.X_observer
        X_earth = self.X_earth
        X_unit = hp.pix2vec(nside, np.arange(npix := hp.nside2npix(nside)))

        emission = np.zeros(npix)
        for comp_name, comp in self.model.components.items():
            integration_config = self.integ[comp_name]

            emissivity = self.model.emissivities.get_emissivity(comp_name, freq)
            comp_emission = comp.get_emission(
                freq, X_observer, X_earth, X_unit, integration_config.R
            )
            comp_emission *= emissivity

            emission += integration_config.integrator(
                comp_emission, 
                integration_config.R, 
                dx=integration_config.dR, 
                axis=0
            )
        
        emission = _coordinates.change_coordinate_system(emission, coord)
        
        return emission