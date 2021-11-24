from typing import Dict, Optional, Union

import numpy as np

from zodipy._coordinates import to_frame, get_target_coordinates, EpochsType
from zodipy._emissivities import get_emissivities
from zodipy._integration_config import integration_config_registry
from zodipy._simulation import instantaneous_emission, time_ordered_emission
from zodipy.models import model_registry


class InterplanetaryDustModel:
    """The Zodipy simulation interface."""

    def __init__(self, model: str = "K98") -> None:
        """Initializes the interface given an interplaneteray dust model.

        By default, the Interplanetary Dust Model used by Zodipy is the
        Kelsall et al. (1998) Interplanetary Dust Model, which includes
        five Zodiacal components:
            - The Diffuse Cloud
            - Three Asteroidal Bands
            - The Circumsolar Ring
            - The Earth-trailing Feature

        The Kelsall model yields the Zodiacal Emission given purely by the
        parametric model, and does not use the emissivity factors fitted by
        the Planck Collaboration.

        Other available models are:
            - Planck13 (all five Kelsall components + emissivity fits for each)
            - Planck15 (cloud + bands + new emissivity fits)
            - Planck18 (cloud + bands + the latest emissivity fits)

        Parameters
        ----------
        model
            The name of a implemented Interplanetary Dust Model.
        """

        self.model = model_registry.get_model(model)
        integration_config = integration_config_registry.get_config("default")
        self.line_of_sights = [
            line_of_sight for line_of_sight in integration_config.values()
        ]

    def get_instantaneous_emission(
        self,
        nside: int,
        freq: float,
        observer: str = "L2",
        epochs: Optional[EpochsType] = None,
        coord: str = "E",
        return_comps: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the instantaneous Zodiacal Emission [MJy/sr].

        By instantaneous emission we mean the emission observed in an instant
        in time.

        The observer location (and optionally the location of the Earth if
        either of the Feature or the Ring components are included) are queried
        from the Horizons JPL ephemerides, given some epoch.

        NOTE: This function returns the fullsky emission from one coordinate
        in space. This means that we evaluate line-of-sights looking in towards
        the inner Solar System where the dust density increases exponentially.
        These are line-of-sights unlikely to be observed by an actual
        experiment.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency at which to evaluate the Zodiacal emission in units of GHz.
        observer
            The observer. Defaults to 'L2'.
        epochs
            The observeration times given as a single epoch, or a list of epochs
            in JD or MJD format, or a dictionary defining a range of times and
            dates; the range dictionary has to be of the form
            {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]',
            'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time
            is used in UTC.
        coord
            Coordinate frame of the output map. Defaults to 'E', which is
            ecliptic coordinates.
        return_comps
            If True, the emission of each component in the model is
            returned individually in a dictionary. Defaults to False.

        Returns
        -------
        emission
            Simulated instantaneous Zodiacal emission in units of MJy/sr.
        """

        observer_coordinates = get_target_coordinates(observer, epochs)
        if self.model.includes_earth_neighboring_components:
            earth_coordinates = get_target_coordinates("earth", epochs)
        else:
            earth_coordinates = np.zeros(3)

        emissivities = get_emissivities(
            freq=freq,
            emissivity=self.model.emissivities,
            components=list(self.model.components.keys()),
        )

        emission = instantaneous_emission(
            nside=nside,
            freq=freq,
            components=list(self.model.components.values()),
            emissivities=emissivities,
            observer_coords=observer_coordinates,
            earth_coords=earth_coordinates,
            line_of_sights=self.line_of_sights,
        )
        if coord != "E":
            emission = to_frame(emission, coord)

        if not return_comps:
            return emission.sum(axis=0)

        return {
            comp.value: emission[idx] for idx, comp in enumerate(self.model.components)
        }

    def get_time_ordered_emission(
        self,
        nside: int,
        freq: float,
        pixels: np.ndarray,
        observer_coordinates: np.ndarray,
        earth_coordinates: Optional[np.ndarray] = None,
        return_comps: bool = False,
        bin: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the Zodiacal emission timestream [MJy/sr].

        Given a timestream of pixels, this function evaluates the Zodiacal
        Emission given a position in space (`observer_coordinates`), and
        optionally Earth's coordinates (`earth_coordinates`) if the
        interplanetary dust model includes either of the Feature or Ring
        components.

        If the bin parameter is set to True, the timestream is binned into a
        HEALPIX map instead.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency at which to evaluate the Zodiacal emission in units of
            GHz.
        pixels
            Chunk of time-ordered pixels corresponding to a parts of a scanning
            strategy.
        observer_coordinates
            The heliocentric coordinates with shape (3,) of the observer over
            the tod chunk.
        earth_coordinates
            The heliocentric coordinates with shape (3,) of the Earth over the
            tod chunk.
        return_comps
            If True, the emission of each component in the model is
            returned individually in a dictionary. Defaults to False.
        bin
            If True, return a binned HEALPIX map of the emission. If False, the
            timestream is returned.

        Returns
        -------
        emission
            Zodiacal emission [MJy/sr] over a timestream of pixels, or the
            binned Zodiacal emission map if bin is set to True.
        """

        if earth_coordinates is None:
            if self.model.includes_earth_neighboring_components:
                raise ValueError(
                    "The Interplanetary Dust model includes earth-neighboring "
                    "components, but no earth_coordinates were given."
                )
            earth_coordinates = np.zeros(3)

        emissivities = get_emissivities(
            freq=freq,
            emissivity=self.model.emissivities,
            components=list(self.model.components.keys()),
        )

        emission = time_ordered_emission(
            nside=nside,
            freq=freq,
            components=list(self.model.components.values()),
            emissivities=emissivities,
            line_of_sights=self.line_of_sights,
            observer_coordinates=observer_coordinates,
            earth_coordinates=earth_coordinates,
            pixel_chunk=pixels,
            bin=bin,
        )

        if not return_comps:
            return emission.sum(axis=0)

        return {
            comp.value: emission[idx] for idx, comp in enumerate(self.model.components)
        }