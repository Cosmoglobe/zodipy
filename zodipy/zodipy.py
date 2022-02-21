from typing import Any, Dict, Optional, Union

import astropy.units as u
from astropy.units import Quantity
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._astroquery import query_target_positions, Epoch
from zodipy._brightness_integral import trapezoidal
from zodipy._dirbe import DIRBE_BAND_REF_WAVELENS, read_color_corr
from zodipy._integration_config import integration_config_registry
from zodipy._interp import interp_source_parameters
from zodipy._model import InterplanetaryDustModel
from zodipy.models import model_registry


class Zodipy:
    """The Zodipy simulation interface.

    Zodipy allows the user to produce simulated timestreams or instantaneous
    maps of the Interplanetary Dust emission. The IPD model implemented is the
    Kelsall et al. (1998) model.

    Methods
    -------
    get_time_oredered_emission
    get_instantaneous_emission
    """

    def __init__(self, model: str = "DIRBE") -> None:
        """Initializes the interface given a model.

        Parameters
        ----------
        model
            The name of the model to initialize. Defaults to DIRBE. See all
            available models in `zodipy.MODELS`.
        """

        self._model = model_registry.get_model(model)
        self._line_of_sights = integration_config_registry.get_config()

    @property
    def info(self) -> str:
        """Returns the string representation of the IPD model."""

        return str(self._model)

    @property
    def parameters(self) -> InterplanetaryDustModel:
        """Returns the IPD model."""

        return self._model

    @u.quantity_input
    def get_time_ordered_emission(
        self,
        freq: Union[Quantity[u.Hz], Quantity[u.m]],
        *,
        nside: int,
        pixels: NDArray[np.int64],
        observer_pos: Quantity[u.AU],
        earth_pos: Optional[Quantity[u.AU]] = None,
        color_corr: bool = False,
        bin: bool = False,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> Quantity[u.MJy / u.sr]:
        """Returns the timestream (or the binned map) of the simulated Zodiacal emission.

        Parameters
        ----------
        freq
            Frequency (or wavelength) at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter of the returned emission map.
        pixels
            Sequence of pixels observed while the observer was at the position
            given by `observer_pos`.
        observer_pos
            The heliocentric ecliptic cartesian position of the observer.
        earth_pos
            The heliocentric ecliptic cartesian position of the Earth. If None,
            the same position is used for the Earth and the observer. NOTE: The
            further away from the Earth the observer gets, the worse this
            assumption becomes. Defaults to None.
        color_corr
            If True, the DIRBE color correction is applied to the thermal
            contribution of the emission. This is usefull for comparing the
            output with the DIRBE TODs or maps.
        bin
            If True, the timestream is binned into a HEALPIX map. Defaults to
            False. NOTE: The binned map will not be normalized by the number of
            hits.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentric
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated timestream or binned map of Zodiacal emission [MJy/sr].
        """

        # Zodipy uses frequency convention.
        freq = freq.to("GHz", equivalencies=u.spectral())

        IPDmodel = self._model

        source_parameters: Dict[str, Any] = {}
        source_parameters["T_0"] = IPDmodel.source_parameters["T_0"]
        source_parameters["delta"] = IPDmodel.source_parameters["delta"]
        source_parameters["color_table"] = None

        if color_corr:
            # Get the DIRBE color correctoin factor.
            wavelength = freq.to("micron", equivalencies=u.spectral())
            try:
                band = DIRBE_BAND_REF_WAVELENS.index(wavelength.value) + 1
            except IndexError:
                raise IndexError(
                    "Color correction is only supported at central DIRBE "
                    "wavelenghts."
                )
            source_parameters["color_table"] = read_color_corr(band=band)

        earth_position = observer_pos if earth_pos is None else earth_pos
        # We reshape the coordinates to broadcastable shapes.
        if earth_position.ndim == 1:
            earth_position = np.expand_dims(earth_position, axis=1)
        if observer_pos.ndim == 1:
            observer_pos = np.expand_dims(observer_pos, axis=1)

        if bin:
            unique_pixels, counts = np.unique(pixels, return_counts=True)
            unit_vectors = _get_rotated_unit_vectors(
                nside=nside, pixels=unique_pixels, coord_out=coord_out
            )

            emission = np.zeros((IPDmodel.ncomps, hp.nside2npix(nside)))
            for idx, (label, component) in enumerate(IPDmodel.components.items()):
                comp_source_params = interp_source_parameters(
                    freq=freq, model=IPDmodel, component=label
                )
                source_parameters.update(comp_source_params)

                emission[idx, unique_pixels] = trapezoidal(
                    component=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=observer_pos.value,
                    earth_pos=earth_position.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=IPDmodel.components[label.CLOUD].X_0,
                    source_parameters=source_parameters,
                )

            emission[:, unique_pixels] *= counts

        else:
            unique_pixels, indicies = np.unique(pixels, return_inverse=True)
            unit_vectors = _get_rotated_unit_vectors(
                nside=nside, pixels=unique_pixels, coord_out=coord_out
            )

            emission = np.zeros((IPDmodel.ncomps, len(pixels)))
            for idx, (label, component) in enumerate(IPDmodel.components.items()):
                comp_source_params = interp_source_parameters(
                    freq=freq, model=IPDmodel, component=label
                )
                source_parameters.update(comp_source_params)

                integrated_comp_emission = trapezoidal(
                    component=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=observer_pos.value,
                    earth_pos=earth_position.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=IPDmodel.components[label.CLOUD].X_0,
                    source_parameters=source_parameters,
                )

                # We map the unique pixel hits back to the timestream
                emission[idx] = integrated_comp_emission[indicies]

        # The output unit is [W / Hz / m^2 / sr] which we convert to [MJy/sr]
        emission = (emission << (u.W / u.Hz / u.m ** 2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    @u.quantity_input
    def get_instantaneous_emission(
        self,
        freq: Union[Quantity[u.Hz], Quantity[u.m]],
        *,
        nside: int,
        observer: str = "L2",
        epochs: Optional[Epoch] = None,
        color_corr: bool = False,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> Quantity[u.MJy / u.sr]:
        """Returns the simulated instantaneous Zodiacal emission.

        The location of the observer are queried from the Horizons JPL
        ephemerides using `astroquery`, given an epoch defined by the `epochs`
        parameter.

        NOTE: This function returns the fullsky emission at a single time. This
        implies that line-of-sights that look directly in towards the Sun is
        included in the emission.

        Parameters
        ----------
        freq
            Frequency (or wavelength) at which to evaluate the Zodiacal emission.
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
        color_corr
            If True, the DIRBE color correction is applied to the thermal
            contribution of the emission. This is usefull for comparing the
            output with the DIRBE TODs or maps.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocenteric
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated (mean) instantaneous Zodiacal emission [MJy/sr].
        """

        # Zodipy uses frequency convention.
        freq = freq.to("GHz", equivalencies=u.spectral())

        IPDmodel = self._model

        source_parameters: Dict[str, Any] = {}
        source_parameters["T_0"] = IPDmodel.source_parameters["T_0"]
        source_parameters["delta"] = IPDmodel.source_parameters["delta"]
        source_parameters["color_table"] = None

        if color_corr:
            # Get the DIRBE color correctoin factor.
            wavelength = freq.to("micron", equivalencies=u.spectral())
            try:
                band = DIRBE_BAND_REF_WAVELENS.index(wavelength.value) + 1
            except IndexError:
                raise IndexError(
                    "Color correction is only supported at central DIRBE "
                    "wavelenghts."
                )
            source_parameters["color_table"] = read_color_corr(band=band)

        observer_positions = [query_target_positions(observer, epochs)]
        if IPDmodel.includes_ring:
            earth_positions = [query_target_positions("earth", epochs)]
        else:
            earth_positions = observer_positions.copy()

        npix = hp.nside2npix(nside)
        unit_vectors = _get_rotated_unit_vectors(
            nside=nside,
            pixels=np.arange(npix),
            coord_out=coord_out,
        )


        emission = np.zeros((IPDmodel.ncomps, npix))
        for observer_position, earth_position in zip(
            observer_positions, earth_positions
        ):
            for idx, (label, component) in enumerate(IPDmodel.components.items()):
                comp_source_params = interp_source_parameters(
                    freq=freq, model=IPDmodel, component=label
                )
                source_parameters.update(comp_source_params)

                integrated_comp_emission = trapezoidal(
                    component=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=observer_position.value,
                    earth_pos=earth_position.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=IPDmodel.components[label.CLOUD].X_0,
                    source_parameters=source_parameters,
                )

                emission[idx] += integrated_comp_emission

        # Averaging the maps
        emission /= len(observer_positions)

        # The output unit is [W / Hz / m^2 / sr] which we convert to [MJy/sr]
        emission = (emission << (u.W / u.Hz / u.m ** 2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)


def _get_rotated_unit_vectors(
    nside: int, pixels: NDArray[np.int64], coord_out: str
) -> NDArray[np.float64]:
    """Returns the unit vectors of a HEALPIX map given a requested output coordinate system.

    Since the Interplanetary Dust Model is evaluated in Ecliptic coordinates,
    we need to rotate any unit vectors defined in another coordinate frame to
    ecliptic before evaluating the model.
    """

    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))
    if coord_out != "E":
        unit_vectors = np.asarray(
            hp.rotator.Rotator(coord=[coord_out, "E"])(unit_vectors)
        )
    return unit_vectors
