from typing import Any, Dict, Literal, Optional, Sequence, Union

import astropy.units as u
from astropy.units import Quantity
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._astroquery import query_target_positions, Epoch
from zodipy._integral import trapezoidal
from zodipy._dirbe import DIRBE_BAND_REF_WAVELENS, read_color_corr
from zodipy._los_config import integration_config_registry
from zodipy._interp import interp_comp_spectral_params
from zodipy._model import InterplanetaryDustModel
from zodipy.models import model_registry
from zodipy._labels import CompLabel


class Zodipy:
    """The Zodipy simulation interface.

    Zodipy allows the user to produce simulated timestreams or instantaneous
    maps of the Interplanetary Dust emission. The IPD model implemented is the
    Kelsall et al. (1998) model.
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
    def get_emission(
        self,
        freq: Union[Quantity[u.Hz], Quantity[u.m]],
        obs_pos: Quantity[u.AU],
        *,
        earth_pos: Optional[Quantity[u.AU]] = None,
        pixels: Optional[Sequence[int]] = None,
        theta: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        phi: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        nside: Optional[int] = None,
        lonlat: bool = False,
        binned: bool = False,
        return_comps: bool = False,
        coord_out: Literal["E", "G", "C"] = "E",
        dirbe_colorcorr: bool = False,
    ) -> Quantity[u.MJy / u.sr]:
        """Simulates and returns a timestream of Zodiacal Emission.

        This function takes in an observer position (`observer_pos`) and a
        sequence of pointings, either in the form of angles on the sky
        (`theta`, `phi`), or as HEALPIX pixels at some resolution
        (`pixels`, `nside`). It then returns the emission an experiment would
        observe from that position as a timestream.

        Parameters
        ----------
        freq
            Frequency (or wavelength) at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or m.
        obs_pos
            The heliocentric ecliptic cartesian position of the observer in AU.
        earth_pos
            The heliocentric ecliptic cartesian position of the Earth in AU. If 
            None, the same position is used for the Earth and the observer. This 
            approximation is only valid for observers close to the Earth. For 
            far-away observers, the circum-solar
            to resolve the Circum-solar ring and Earth-trailing Feature for observer
            far from the Earth. Defaults to None.
        pixels
            Sequence of observed pixels while the observer was at the position
            given by `observer_pos`.
        theta, phi
            Angular coordinates (co-latitude, longitude) of a point, or a
            sequence of points, on the sphere. Units must be radians or degrees.
        nside
            HEALPIX map resolution parameter of the pixels (and optionally the
            binned output map). Must be specified if `pixels` is provided or if
            `binned` is set to True.
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        binned
            If True, the timestream is binned into a HEALPIX map. Defaults to
            False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentric
            ecliptic coordinates).
        dirbe_colorcorr
            If True, the DIRBE color correction is applied to the thermal
            contribution of the emission. This is usefull for comparing the
            output with the DIRBE TODs or maps.

        Returns
        -------
        emission
            Simulated timestream or binned map of Zodiacal emission [MJy/sr].
        """

        if pixels is not None:
            if nside is None:
                raise ValueError(
                    "get_time_ordered_emission() got an argument for 'pixels', "
                    "but argument 'nside' is missing"
                )
            if np.size(pixels) == 1:
                pixels = np.expand_dims(pixels, axis=0)

        if binned and nside is None:
            raise ValueError(
                "get_time_ordered_emission() has 'binned' set to True, but "
                "argument 'nside' is missing"
            )

        if (theta is not None and phi is None) or (phi is not None and theta is None):
            raise ValueError(
                "get_time_ordered_emission() is missing an argument for either "
                "'phi' or 'theta'"
            )

        if theta is not None and phi is not None:
            if phi.size == 1:
                phi = np.expand_dims(phi, axis=0)
            if theta.size == 1:
                theta = np.expand_dims(theta, axis=0)

        if pixels is not None and (phi is not None or theta is not None):
            raise ValueError(
                "get_time_ordered_emission() got an argument for both 'pixels'"
                "and 'theta' or 'phi' but can only use one of the two"
            )

        if lonlat:
            if pixels is not None:
                raise ValueError(
                    "get_time_ordered_emission() has 'lonlat' set to True "
                    "but 'theta' and 'phi' is not given"
                )
            theta = theta.to(u.deg)
            phi = phi.to(u.deg)

        freq = freq.to("GHz", equivalencies=u.spectral())

        params = model.source_params.copy()
        model = self._model
        if dirbe_colorcorr:
            # Get the DIRBE color correctoin factor.
            lambda_ = freq.to("micron", equivalencies=u.spectral())
            try:
                band = DIRBE_BAND_REF_WAVELENS.index(lambda_.value) + 1
            except IndexError:
                raise IndexError(
                    "Color correction is only supported at central DIRBE "
                    "wavelenghts."
                )
            params["color_table"] = read_color_corr(band=band)
        else:
            params["color_table"] = None

        earth_pos = obs_pos if earth_pos is None else earth_pos

        # We reshape the coordinates to broadcastable shapes.
        obs_pos = obs_pos.reshape(3, 1)
        earth_pos = earth_pos.reshape(3, 1)

        cloud_offset = model.comps[CompLabel.CLOUD].X_0

        if binned:
            if pixels is not None:
                unique_pixels, counts = np.unique(pixels, return_counts=True)
                unit_vectors = _rotated_pix2vec(
                    pixels=unique_pixels,
                    coord_out=coord_out,
                    nside=nside,
                )

            else:
                unique_angles, counts = np.unique(
                    np.asarray([theta, phi]), return_counts=True, axis=1
                )
                unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
                unit_vectors = _rotated_ang2vec(
                    *unique_angles,
                    lonlat=lonlat,
                    coord_out=coord_out,
                )

            emission = np.zeros((model.ncomps, hp.nside2npix(nside)))

            for idx, (label, component) in enumerate(model.comps.items()):
                comp_spectral_params = interp_comp_spectral_params(
                    comp_label=label,
                    freq=freq,
                    spectral_params=model.spectral_params,
                )
                params.update(comp_spectral_params)

                emission[idx, unique_pixels] = trapezoidal(
                    comp=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=obs_pos.value,
                    earth_pos=earth_pos.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=cloud_offset,
                    source_params=params,
                )

            emission[:, unique_pixels] *= counts

        else:
            if pixels is not None:
                unique_pixels, indicies = np.unique(pixels, return_inverse=True)
                unit_vectors = _rotated_pix2vec(
                    pixels=unique_pixels,
                    coord_out=coord_out,
                    nside=nside,
                )
                emission = np.zeros((model.ncomps, len(pixels)))

            else:
                unique_angles, indicies = np.unique(
                    np.asarray([theta, phi]), return_inverse=True, axis=1
                )
                unit_vectors = _rotated_ang2vec(
                    *unique_angles,
                    lonlat=lonlat,
                    coord_out=coord_out,
                )
                emission = np.zeros((model.ncomps, len(phi)))

            for idx, (label, component) in enumerate(model.comps.items()):
                comp_spectral_params = interp_comp_spectral_params(
                    comp_label=label,
                    freq=freq,
                    spectral_params=model.spectral_params,
                )
                params.update(comp_spectral_params)

                integrated_comp_emission = trapezoidal(
                    comp=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=obs_pos.value,
                    earth_pos=earth_pos.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=cloud_offset,
                    source_params=params,
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

        model = self._model

        params: Dict[str, Any] = model.source_params.copy()
        if color_corr:
            # Get the DIRBE color correctoin factor.
            lambda_ = freq.to("micron", equivalencies=u.spectral())
            try:
                band = DIRBE_BAND_REF_WAVELENS.index(lambda_.value) + 1
            except IndexError:
                raise IndexError(
                    "Color correction is only supported at central DIRBE "
                    "wavelenghts."
                )
            params["color_table"] = read_color_corr(band=band)
        else:
            params["color_table"] = None

        observer_positions = query_target_positions(observer, epochs)
        if model.includes_ring:
            earth_positions = query_target_positions("earth", epochs)
        else:
            earth_positions = observer_positions.copy()

        cloud_offset = model.comps[CompLabel.CLOUD].X_0

        npix = hp.nside2npix(nside)
        unit_vectors = _rotated_pix2vec(
            nside=nside,
            pixels=np.arange(npix),
            coord_out=coord_out,
        )

        emission = np.zeros((model.ncomps, npix))
        for observer_position, earth_position in zip(
            observer_positions, earth_positions
        ):
            for idx, (label, component) in enumerate(model.comps.items()):
                comp_spectral_params = interp_comp_spectral_params(
                    comp_label=label,
                    freq=freq,
                    spectral_params=model.spectral_params,
                )
                params.update(comp_spectral_params)

                integrated_comp_emission = trapezoidal(
                    comp=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=observer_position.value,
                    earth_pos=earth_position.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=cloud_offset,
                    source_params=params,
                )

                emission[idx] += integrated_comp_emission

        # Averaging the maps
        emission /= len(observer_positions)

        # The output unit is [W / Hz / m^2 / sr] which we convert to [MJy/sr]
        emission = (emission << (u.W / u.Hz / u.m ** 2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)


def _rotated_pix2vec(
    pixels: Sequence[int], nside: int, coord_out: str
) -> NDArray[np.float64]:
    """Returns rotated unit vetors from pixels.

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


def _rotated_ang2vec(
    phi: Union[Quantity[u.rad], Quantity[u.deg]],
    theta: Union[Quantity[u.rad], Quantity[u.deg]],
    lonlat: bool,
    coord_out: str,
) -> NDArray[np.float64]:
    """Returns rotated unit vectors from angles.

    Since the Interplanetary Dust Model is evaluated in Ecliptic coordinates,
    we need to rotate any unit vectors defined in another coordinate frame to
    ecliptic before evaluating the model.
    """

    unit_vectors = np.asarray(hp.ang2vec(phi, theta, lonlat=lonlat)).transpose()
    if coord_out != "E":
        unit_vectors = np.asarray(
            hp.rotator.Rotator(coord=[coord_out, "E"])(unit_vectors)
        )
    return unit_vectors
