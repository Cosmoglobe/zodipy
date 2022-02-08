from typing import Dict, Optional, Sequence, Union

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._astroquery import query_target_positions
from zodipy._bandpass import DIRBE_BAND_REF_WAVELENS, read_color_corr
from zodipy._integration_config import integration_config_registry
from zodipy._labels import LABEL_TO_CLASS
import zodipy._simulation as simulation
from zodipy.models import model_registry


class Zodipy:
    """The Zodipy simulation interface.

    Zodipy implements the Kelsall et al. (1998) Interplanetary Dust Model,
    which includes five (six) Zodiacal components:
        - The Diffuse Cloud (cloud)
        - Three Asteroidal Bands (band1, band2, band3)
        - The Circumsolar Ring (ring) + The Earth-trailing Feature (feature)
    """

    def __init__(self, model: str = "DIRBE") -> None:
        """Initializes the interface given a model (fitted source parameters).

        Parameters
        ----------
        model
            The name of the model to initialize. Defaults to DIRBE. Other
            possible models are: Planck13, Planck15, and, Planck18.
        """

        self.model = model_registry.get_model(model)
        self.line_of_sights = integration_config_registry.get_config("default")

    @u.quantity_input(freq_or_wavelength=("Hz", "m", "micron"))
    def get_instantaneous_emission(
        self,
        freq_or_wavelen: u.Quantity,
        *,
        nside: int,
        observer: str = "L2",
        epochs: Optional[Union[float, Sequence[float], Dict[str, str]]] = None,
        color_corr: bool = True,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> NDArray[np.float64]:
        """Simulates and returns the instantaneous Zodiacal Emission [MJy/sr].

        By instantaneous emission we mean the emission observed at an instant
        in time. If multiple epochs are given, the returned emission will be
        the mean of all simulated instantaneous observations.

        The location of the observer, and optionally, the Earth, are queried
        from the Horizons JPL ephemerides using `astroquery`, given an epoch
        defined by the `epochs` parameter.

        NOTE: This function returns the fullsky emission at a single time. This
        means that line-of-sights that sometimes points directly towards the
        inner Solar System and through the Sun, where the dust density
        increases exponentially are evaluated. Such line-of-sights are unlikely
        to be observed by an actual observer, and as such, the simulated
        emission will appear very bright in these npregions.

        Parameters
        ----------
        freq_or_wavelen
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter of the returned emission map.
        observer
            The name of the observer for which we quiery its location given
            the `epochs` parameter. Defaults to 'L2'.
        epochs
            The observeration times given as a single epoch, or a list of epochs
            in JD or MJD format, or a dictionary defining a range of times and
            dates; the range dictionary has to be of the form
            {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]',
            'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time
            is used in UTC.
        return_comps
            If True, the emission is returned component-wise in a dictionary.
            Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentered
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated (mean) instantaneous Zodiacal emission [MJy/sr].
        """

        color_table: Optional[NDArray[np.float64]]
        if color_corr:
            wavelen = freq_or_wavelen.to("micron", equivalencies=u.spectral())
            band = DIRBE_BAND_REF_WAVELENS.index(wavelen.value) + 1
            color_table = read_color_corr(band=band)
        else:
            color_table = None
        freq = freq_or_wavelen.to("GHz", equivalencies=u.spectral())

        observer_positions = query_target_positions(observer, epochs)
        if self.model.includes_ring:
            earth_positions = query_target_positions("earth", epochs)
        else:
            earth_positions = observer_positions.copy()

        emission = simulation.instantaneous_emission(
            nside=nside,
            freq=freq.value,
            model=self.model,
            line_of_sights=self.line_of_sights,
            observer_pos=observer_positions,
            earth_pos=earth_positions,
            color_table=color_table,
            coord_out=coord_out,
        )

        return emission if return_comps else emission.sum(axis=0)

    @u.quantity_input(freq_or_wavelen=("Hz", "m", "micron"))
    def get_time_ordered_emission(
        self,
        freq_or_wavelen: u.Quantity,
        *,
        nside: int,
        pixels: NDArray[np.int64],
        observer_pos: NDArray[np.float64],
        earth_pos: Optional[NDArray[np.float64]] = None,
        color_corr: bool = True,
        bin: bool = False,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> NDArray[np.float64]:
        """Simulates and returns the Zodiacal emission as a timestream or a map
        in units of MJy/sr.

        Given a chunk of pixels given by `pixels` observed from a location given
        by `observer_pos`, the Zodiacal emission is computed through a per-pixel
        line-of-sight evaluation.

        Parameters
        ----------
        freq_or_wavelength
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter of the returned emission map.
        pixels
            Sequence of pixels observed while the observer was at the position
            given by `observer_pos`.
        observer_pos
            The heliocentric ecliptic cartesian position of the observer.
        earth_pos
            The heliocentric ecliptic cartesian position of the Earth. If None,
            the same position is used for the Earth and the observer. Defaults
            to None.
        color_corr
            If True, the DIRBE color correction factors are used when evaluating
            the model. NOTE: the frequency must match on of the central
            frequencies of the DIRBE bands.
        bin
            If True, the output is a HEALPIX map instead of a timestream.
            Defaults to False.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentric
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated timestream or map of Zodiacal emission in units of MJy/sr.
        """

        color_table: Optional[NDArray[np.float64]]
        if color_corr:
            wavelen = freq_or_wavelen.to("micron", equivalencies=u.spectral())
            band = DIRBE_BAND_REF_WAVELENS.index(wavelen.value) + 1
            color_table = read_color_corr(band=band)
        else:
            color_table = None
        freq = freq_or_wavelen.to("GHz", equivalencies=u.spectral())

        if earth_pos is None:
            earth_pos = observer_pos

        emission = simulation.time_ordered_emission(
            model=self.model,
            nside=nside,
            freq=freq.value,
            line_of_sights=self.line_of_sights,
            observer_pos=observer_pos,
            earth_pos=earth_pos,
            pixel_chunk=pixels,
            color_table=color_table,
            bin=bin,
            coord_out=coord_out,
        )

        return emission if return_comps else emission.sum(axis=0)

    def __str__(self) -> str:
        """String representation of the Interplanetary dust model used."""

        reprs = []
        for label in self.model.components:
            component_class = LABEL_TO_CLASS[label]
            component_repr = f"{component_class.__name__}" + "\n"
            reprs.append(f"({label.value}): {component_repr}")

        main_repr = "InterplanetaryDustModel("
        main_repr += f"\n  name: {self.model.name}"
        main_repr += "\n  components( "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += "  )"
        main_repr += "\n)"

        return main_repr
