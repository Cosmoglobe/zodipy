from typing import Optional, Union
from datetime import datetime

import astropy.units as u
import healpy as hp
import numpy as np

from zodipy import models
from zodipy import _coordinates
from zodipy import _integration as integ


class Zodi:
    """Interface for simulating the Zodiacal emission.
    
    The emission is simulated as the instantaneous emission seen by an 
    observer at some location in a region nearby earth at some time.
    """

    def __init__(
        self, 
        observer : Optional[str] = 'L2',
        observation_time : Optional[datetime] = datetime.now().date(),
        earth_position : Optional[np.ndarray] = None,
        model : Optional[models.Model] = models.PLANCK_2018,
        integ : Optional[integ.IntegrationConfig] = integ.DEFAULT,
    ) -> None:
        """Initializing the zodi interface.

        The geometric setup of the simulation, the model parameters and 
        components, and the integration configuration used when integrating
        up the emission are all configured here in the initialization.
        
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

        Methods
        -------
        simulate
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
        self, 
        nside: int, 
        freq: Union[float, u.Quantity], 
        coord: str = 'G',
        return_comps: bool = False
    ) -> np.ndarray:
        """Simulates the Zodiacal emission given a frequency in MJy/sr.

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
            Simulated Zodiacal emission [MJy/sr] for some nside at some 
            frequency.
        """
        
        if isinstance(freq, u.Quantity):
            freq = freq.to('GHz').value

        X_observer = self.X_observer
        X_earth = self.X_earth
        X_unit = hp.pix2vec(nside, np.arange(npix := hp.nside2npix(nside)))

        if return_comps:
            emission = np.zeros((len(self.model.components), npix))
        else:
            emission = np.zeros(npix)

        for idx, (comp_name, comp) in enumerate(self.model.components.items()):
            integration_config = self.integ[comp_name]
            comp_emission = comp.get_emission(
                freq, X_observer, X_earth, X_unit, integration_config.R
            )
            integrated_comp_emission = integration_config.integrator(
                comp_emission, 
                integration_config.R, 
                dx=integration_config.dR, 
                axis=0
            )

            comp_emissivity = self.model.emissivities.get_emissivity(comp_name, freq)
            integrated_comp_emission *= comp_emissivity

            if return_comps:
                emission[idx] = integrated_comp_emission
            else: 
                emission += integrated_comp_emission
        
        emission *= 1e20    # Converting to MJy / sr from W / m^2 sr hz

        emission = _coordinates.change_coordinate_system(emission, coord)
        
        return emission