from typing import Dict, Optional, Union

import numpy as np

from zodipy._coordinates import to_frame, get_target_coordinates, EpochsType
from zodipy._emissivities import get_emissivities
from zodipy._integration_config import integration_config_registry
from zodipy._simulation import instantaneous_emission, time_ordered_emission
from zodipy.models import model_registry


class InterplanetaryDustModel:
    """The Zodipy simulation interface."""

    def __init__(self, model: str = "Planck18") -> None:
        """Initializes a interplaneteray dust model.

        Parameters
        ----------
        model
            The name of a implemented interplanetary dust model.
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
        """Simulates and returns the instantaneous Zodiacal Emission.

        The observer location (and optionally the location of the Earth) are
        quiried from the Horizons JPL ephemerides given some epoch.

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
        pixel_chunk: np.ndarray,
        observer_coordinates: np.ndarray,
        earth_coordinates: Optional[np.ndarray] = None,
        return_comps: bool = False,
        bin: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the Zodiacal emission [MJy/sr] in a timestream.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency at which to evaluate the Zodiacal emission in units of
            GHz.
        pixel_chunk
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
        emission_timestream
            Timestream of the Zodiacal emission [MJy/sr] over a chunk of
            time-ordered pixels.
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

        emission_timestream = time_ordered_emission(
            nside=nside,
            freq=freq,
            components=list(self.model.components.values()),
            emissivities=emissivities,
            line_of_sights=self.line_of_sights,
            observer_coordinates=observer_coordinates,
            earth_coordinates=earth_coordinates,
            pixel_chunk=pixel_chunk,
            bin=bin,
        )

        if not return_comps:
            return emission_timestream.sum(axis=0)

        return {
            comp.value: emission_timestream[idx]
            for idx, comp in enumerate(self.model.components)
        }