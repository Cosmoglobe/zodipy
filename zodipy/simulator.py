from datetime import datetime
from astropy.coordinates.solar_system import get_body
import healpy as hp
import numpy as np
import datetime

from astropy import time
from astropy import coordinates

from zodipy.data import models
from zodipy import integration
from zodipy import functions as F


now = time.Time(datetime.datetime.now())
with coordinates.solar_system_ephemeris.set('builtin'):
    earth_position = coordinates.get_body('earth', now)
    earth_position = earth_position.transform_to(
        coordinates.HeliocentricMeanEcliptic
    ).cartesian

print(earth_position)
class Simulator:
    def __init__(
        self, 
        nside : int,
        earth_position : None,
        observer_position : tuple = None,
        model : models.Model = models.PLANCK_2013,
        integration_config : integration.IntegrationConfig = integration.DEFAULT_CONFIG,
    ) -> None:
        """Initializing the Zodi interface."""

        self.nside = nside 
        self.observer_position = observer_position

        if observer_position is None:
            self.observer_position = np.array([[1],[0],[0]])

        self.components = model.components
        self.emissivities = model.emissivities
        self.integration_config = integration_config

    @staticmethod
    def get_pixel_unit_vectors(nside):
        """Returns unit vectors for a map of a given nside."""

        return np.asarray(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))  

    def get_emissivity(self, freq, component):
        """Returns the interpolated emissivity given a frequency."""

        freqs = self.emissivities['freqs']
        emissivities = self.emissivities[component]

        return np.interp(freq, freqs, emissivities)

    @staticmethod
    def get_blackbody_emission(R, freq):
        """Returns the blackbody emission for a single dust particle.
        
        Parameters
        ----------
        R : `np.ndarray`
            Heliocentric distance.
        freq : float
            Frequency at which to evaluate the blackbody.

        Returns
        -------
        `np.ndarray`
            Blackbody emission for a single dust particle.
        """

        T = F.interplanetary_temperature(R)
        return F.blackbody_emission(T, freq)

    def get_emission(self, freq):
        """Returns the model emission given a frequency.

        Parameters
        ----------
        freq : `astropy.units.Quantity`
            Frequency at which to evaluate the IPD model [Hz].
        """

        X_observer = self.observer_position
        X_unit = self.get_pixel_unit_vectors(self.nside)

        NPIX = hp.nside2npix(self.nside)
        emission = np.zeros(NPIX)

        for comp_name, comp in self.components.items():
            emissivity = self.get_emissivity(freq, comp_name)
            integration_config = self.integration_config[comp_name]
            comp_emission = np.zeros((integration_config.n, NPIX))

            for idx, R in enumerate(integration_config.shells):
                prime_coords, R_helio = comp.get_coordinates(X_observer, X_unit, R)
                density = comp.get_density(*prime_coords)
                blackbody_emission = self.get_blackbody_emission(R_helio, freq)
                comp_emission[idx] += emissivity * blackbody_emission * density  

            emission += integration_config.integrator(
                comp_emission, 
                integration_config.shells, 
                dx=integration_config.dshells, 
                axis=0
            )

        return emission