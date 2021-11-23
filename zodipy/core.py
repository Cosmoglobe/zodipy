from typing import Dict, Optional, Union

import numpy as np

from zodipy._coordinates import to_frame, get_target_coordinates, Epochs
from zodipy._component_labels import ComponentLabel
from zodipy._emissivities import get_emissivities
from zodipy._integration_config import integration_config_registry
from zodipy._strategies import instantaneous_emission, time_ordered_emission
from zodipy.models import model_registry


class InstantaneousModel:
    """Model for instantaneous interplanetary dust emission simulations.

    This model is used to simulate one, or the average of multiple
    instantaneous simulated observations. Observer coordinates are extracted
    using the JPL Horizons API."""

    def __init__(
        self,
        observer: str = "L2",
        epochs: Optional[Epochs] = None,
        model: str = "Planck18",
        integration_config: str = "default",
    ) -> None:
        """Initializes the IPD model.

        Parameters
        ----------
        observer
            The observer. Defaults to 'L2'.
        epochs
            The observeration times given as a single epoch, or a list of epochs
            in JD or MJD format, or a dictionary defining a range of times and
            dates; the range dictionary has to be of the form
            {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]',
            'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time
            is used in UTC.
        model
            The Interplanetary dust model used in the simulation. Defaults to
            "Planck18".
        integration_config
            Configuration for the line-of-sight integration. Defaults to "default".
        """

        self.model = model_registry.get_model(model)
        config = integration_config_registry.get_config(integration_config)
        self.line_of_sights = [line_of_sight for line_of_sight in config.values()]
        self.observer_coordinates = get_target_coordinates(observer, epochs)
        if (
            ComponentLabel.RING in self.model.components
            or ComponentLabel.FEATURE in self.model.components
        ):
            self.earth_coordinates = get_target_coordinates("earth", epochs)
        else:
            self.earth_coordinates = np.zeros(3)

    def __call__(
        self,
        nside: int,
        freq: float,
        coord: str = "E",
        return_comps: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the Zodiacal emission in units of MJy/sr.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency at which to evaluate the Zodiacal emission in units of GHz.
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

        components = self.model.components
        emissivities = get_emissivities(
            freq=freq,
            emissivity=self.model.emissivities,
            components=list(components.keys()),
        )

        emission = instantaneous_emission(
            nside=nside,
            freq=freq,
            components=list(components.values()),
            emissivities=emissivities,
            observer_coords=self.observer_coordinates,
            earth_coords=self.earth_coordinates,
            line_of_sights=self.line_of_sights,
        )
        if coord != "E":
            emission = to_frame(emission, coord)

        if not return_comps:
            return emission.sum(axis=0)

        return {comp.value: emission[idx] for idx, comp in enumerate(components)}


class TimeOrderedModel:
    """Model for time-ordered interplanetary dust emission simulations.

    This model is used to compute the Zodiacal in the timestream for a
    given scanning strategy defined by the pixel arrays.
    """

    def __init__(
        self,
        model: str = "Planck18",
        integration_config: str = "default",
    ) -> None:
        """Initializes the IPD model.

        Parameters
        ----------
        model
            The Interplanetary dust model used in the simulation. Defaults to
            "Planck18".
        integration_config
            Configuration for the line-of-sight integration. Defaults to "default".
        """

        self.model = model_registry.get_model(model)
        config = integration_config_registry.get_config(integration_config)
        self.line_of_sights = [line_of_sight for line_of_sight in config.values()]

    def __call__(
        self,
        nside: int,
        freq: float,
        pixel_chunk: np.ndarray,
        observer_coordinates: np.ndarray,
        earth_coordinates: np.ndarray,
        return_comps: bool = False,
    ):
        """Simulates and returns the Zodiacal emission timestream in units of MJy/sr.

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
            The heliocentric coordinates of the observer over the tod chunk.
        earth_coordinates
            The heliocentric coordinates of the Earth over the tod chunk.
        return_comps
            If True, the emission of each component in the model is
            returned individually in a dictionary. Defaults to False.

        Returns
        -------
        emission_timestream
            Timestream of the Zodiacal emission in units of MJy/sr over a chunk
            of time-ordered pixels.
        """

        components = self.model.components
        emissivities = get_emissivities(
            freq=freq,
            emissivity=self.model.emissivities,
            components=list(components.keys()),
        )
        emission_timestream = time_ordered_emission(
            pixel_chunk=pixel_chunk,
            freq=freq,
            nside=nside,
            components=list(components.values()),
            emissivities=emissivities,
            line_of_sights=self.line_of_sights,
            observer_coordinates=observer_coordinates,
            earth_coordinates=earth_coordinates,
        )

        if not return_comps:
            return emission_timestream.sum(axis=0)

        return {
            comp.value: emission_timestream[idx] for idx, comp in enumerate(components)
        }


if __name__ == "__main__":
    import healpy as hp
    import matplotlib.pyplot as plt

    epochs = {
        "start": "01-01-2020",
        "stop": "01-01-2021",
        "step": "91d",
    }

    # model = InstantaneousModel(model="K98", epochs=[2459215.50000, 2459238.50000])

    # model = InstantaneousModel(epochs=epochs)

    # hp.mollview(model(128, 800), norm="hist", coord=["E", "G"])
    # plt.show()
    # exit()
    import h5py

    DATA_PATH = "/Users/metinsan/Documents/doktor/data/Phot01.hdf5"

    model = TimeOrderedModel(model="Planck18")
    with h5py.File(DATA_PATH, "r") as file:
        for tod_chunk in file:
            pixels = np.asarray(file[f"000156/A/pix"][()])
            # tods = np.asarray(file[f"{tod_chunk}/A/tod"][()])
            time_stream = model(
                freq=857,
                nside=128,
                pixel_chunk=pixels,
                observer_coordinates=np.asarray([1, 0, 0]),
                earth_coordinates=np.asarray([1, 0, 0]),
                return_comps=True
            )
            # inds = tods > 0
            t = np.arange(len(pixels))
            # fig, axes = plt.subplots(nrows=2, sharex=True)
            # axes[0].plot(t[inds], tods[inds], "k.", ms=1)
            for comp in time_stream:
                plt.plot(t, time_stream[comp], label=comp)

            plt.plot(t, np.sum(list(time_stream.values()),axis=0), label="total")
            plt.legend()
            plt.show()
            exit()
