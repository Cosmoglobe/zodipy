import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

from zodipy.data import models
from zodipy import functions as F


class Zodi:
    def __init__(
        self, 
        nside : int,
        observer_position : tuple = None,
        model : models.Model = models.PLANCK_2018,
        R_min : float = 0.001,
        R_max : float = 30,
        n_R : int = 50,
    ) -> None:
        """Initializing the Zodi interface."""

        self.nside = nside 
        self.observer_position = observer_position

        if observer_position is None:
            self.observer_position = np.array([[1],[0],[0]])

        self.components = model.components
        self.emissivities = model.emissivities

        self.R_min = R_min
        self.R_max = R_max
        self.n_R = n_R

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

        # R_grid is the distance to each shell for which we will evaluate
        # the emission.
        R_grid, dR_grid = np.linspace(
            self.R_min, self.R_max, self.n_R, retstep=True
        )

        X_observer = self.observer_position
        X_unit = self.get_pixel_unit_vectors(self.nside)

        emission = np.zeros((self.n_R, hp.nside2npix(self.nside)))
        for idx, R in enumerate(R_grid):            
            for name, component in self.components.items():
                prime_coords, R_helio = component.get_coordinates(
                    X_observer, X_unit, R
                )
                density = component.get_density(*prime_coords)
                blackbody_emission = self.get_blackbody_emission(R_helio, freq)
                emissivity = self.get_emissivity(freq, name)
                emission[idx] += emissivity * blackbody_emission * density

        emission = np.trapz(emission, R_grid, dx=dR_grid, axis=0)
        return emission




if __name__ == '__main__':
    nside = 128
    zodi = Zodi(nside)
    emission = zodi.get_emission(800)
    hp.mollview(emission, norm='hist', coord='EG')
    plt.show()