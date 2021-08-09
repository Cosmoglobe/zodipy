from typing import Optional
from datetime import datetime

import healpy as hp
import numpy as np

from zodipy import models
from zodipy import _coordinates
from zodipy import integration as integ


class Zodi:
    """Interface for simulations of the interplanetary dust emission."""

    def __init__(
        self, 
        observer : Optional[str] = 'L2',
        observation_time : Optional[datetime] = datetime.now(),
        earth_position : Optional[np.ndarray] = None,
        model : models.Model = models.PLANCK_2018,
        integ : integ.IntegrationConfig = integ.DEFAULT_CONFIG,
    ) -> None:
        """Initializing the Zodi interface.
        
        Parameters
        ----------
        observer : str, optional
            The observer. Default is L2.
        observation_time : `datetime.datetime`, optional
            The time of the observation. Default is the current time.
        earth_position : `numpy.ndarray`, optional
            Heliocentric coordinates of the Earth. If None, Earth's 
            coordinates from the current time is used. Default is None.
        model : `zodipy.models.Model`, optional
            The Interplanteary dust model used in the simulation. 
            Default is the model used in the Planck 2018 analysis.
        integ : `zodipy.integration.IntegrationConfig`, optional
            Integration config object determining the integration details
            used in the simulation.
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

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """Returns the model emission given a frequency.

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the IPD model [Hz].
        """

        NPIX = hp.nside2npix(nside)
        emission = np.zeros(NPIX)

        X_observer = self.X_observer
        X_earth = self.X_earth
        X_unit = np.asarray(
            hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
        )  

        model = self.model
        for comp_name, comp in model.components.items():
            integration_config = self.integ[comp_name]
            comp_emission = np.zeros((integration_config.n, NPIX))
        
            for idx, R in enumerate(integration_config.R):
                comp_emission[idx] = comp.get_emission(freq, X_observer, X_earth, X_unit, R)

            emissivity = model.emissivities.get_emissivity(comp_name, freq)
            comp_emission *= emissivity
            emission += integration_config.integrator(
                comp_emission, 
                integration_config.R, 
                dx=integration_config.dR, 
                axis=0
            )

        return emission