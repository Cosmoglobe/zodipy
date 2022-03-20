from typing import List, Literal, Optional, Sequence, Tuple, Union

from astropy.coordinates import (
    get_body,
    HeliocentricMeanEcliptic,
    solar_system_ephemeris,
)
import astropy.units as u
from astropy.units import Quantity, quantity_input
from astropy.time import Time
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._integral import trapezoidal
from zodipy._interp import interp_phase_coeffs, interp_comp_spectral_params
from zodipy._los_config import integration_config_registry
from zodipy._model import InterplanetaryDustModel
from zodipy.models import model_registry


__all__ = ("Zodipy",)

# L2 not included in the astropy api so we manually include support for it.
ADDITIONAL_SUPPORTED_OBS = ["semb-l2"]
DISTANCE_EARTH_TO_L2 = 0.009896235034000056 * u.AU


class Zodipy:
    """The Zodipy interface.

    Zodipy simulates the Zodiacal emission that a Solar System observer is
    predicted to see given the Kelsall et al. (1998) interplanetary dust model. 
    """

    def __init__(self, model: str = "DIRBE", ephemeris: str = "de432s") -> None:
        """Initializes Zodipy for a given a model and emphemeris.

        Parameters
        ----------
        model
            The name of the interplanetary dust model. Defaults to DIRBE. See all
            available models with `zodipy.MODELS`.
        ephemeris
            Ephemeris used to compute the positions of the observer and Earth.
            Defaults to 'de432s' which requires downloading (and caching) a 10 
            MB file. For more information on available ephemeridis, please visit
            https://docs.astropy.org/en/stable/coordinates/solarsystem.html
        """

        self._model = model_registry.get_model(model)
        self._line_of_sights = integration_config_registry.get_config()
        solar_system_ephemeris.set(ephemeris)

    @property
    def model(self) -> InterplanetaryDustModel:
        """Returns the IPD model."""

        return self._model

    @property
    def info(self) -> Optional[str]:
        return self._model.meta.get("info")

    @property
    def observers(self) -> List[str]:
        """Returns all observers suported by the ephemeridis."""

        return list(solar_system_ephemeris.bodies) + ADDITIONAL_SUPPORTED_OBS

    @quantity_input
    def get_emission(
        self,
        freq: Union[Quantity[u.Hz], Quantity[u.m]],
        obs_time: Time,
        obs: str = "earth",
        *,
        obs_pos: Optional[Quantity[u.AU]] = None,
        pixels: Optional[Union[int, Sequence[int], NDArray[np.integer]]] = None,
        theta: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        phi: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        nside: Optional[int] = None,
        lonlat: bool = False,
        binned: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
        colorcorr_table: Optional[NDArray[np.floating]] = None,
    ) -> Quantity[u.MJy / u.sr]:
        """Returns simulated Zodiacal Emission.

        This function takes as arguments a frequency or wavelength (`freq`),
        time of observation (`obs_time`), an a Solar System observer (`obs`).
        The position of the observer is computed using the pehemeris specified 
        in the initialization of `Zodipy`. Optionally, the observer position 
        can be explicitly specified with the `obs_pos` argument, which 
        overrides`obs`. The pointing, for which to compute the emission, is 
        specified either in the form of angles on the sky (`theta`, `phi`), or 
        as HEALPIX pixels at some resolution (`pixels`, `nside`).

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or m.
        obs_time
            Time of observation (`astropy.time.Time` object).
        obs
            The solar system observer. A list of all support observers (for a
            given ephemeridis) is specified in `observers` attribute of the
            `zodipy.Zodipy` instance. Defaults to 'earth'.
        obs_pos
            The heliocentric ecliptic cartesian position of the observer in AU.
            Overrides the `obs` argument. Default is None.
        pixels
            A single, or a sequence of HEALPIX pixels representing points on
            the sky. The `nside` parameter, which specifies the resolution of
            these pixels, must also be provided along with this argument.
        theta, phi
            Angular coordinates (co-latitude, longitude ) of a point, or a
            sequence of points, on the sphere. Units must be radians or degrees.
            co-latitude must be in [0, pi] rad, and longitude in [0, 2*pi] rad.
        nside
            HEALPIX map resolution parameter of the pixels (and optionally the
            binned output map). Must be specified if `pixels` is provided or if
            `binned` is set to True.
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
            Seeting lonlat to True corresponds to theta=RA, phi=DEC
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        binned
            If True, the emission is binned into a HEALPIX map with resolution
            given by the `nside` argument in the coordinate frame corresponding to
            `coord_in`. Defaults to False.
        coord_in
            Coordinates frame of the pointing. Assumes 'E' (ecliptic coordinates)
            by default.
        colorcorr_table
            An array of shape (2, n) where the first column is temperatures
            in K, and the second column the corresponding color corrections.
            The color corrections should be for B(lambda, T) and is only applied
            to the thermal contribution of the emission.

        Returns
        -------
        emission
            Sequence of simulated Zodiacal emission in units of 'MJy/sr' for
            each pointing. If the pointing is provided in a time-ordered manner,
            then the output of this function can be interpreted as the observered
            Zodiacal Emission timestream.
        """

        if obs.lower() not in self.observers:
            raise ValueError(
                f"observer {obs!r} not supported by ephemeridis "
                f"{solar_system_ephemeris._value!r} or 'Zodipy'"
            )

        if pixels is not None:
            if phi is not None or theta is not None:
                raise ValueError(
                    "get_time_ordered_emission() got an argument for both 'pixels'"
                    "and 'theta' or 'phi' but can only use one of the two"
                )
            if nside is None:
                raise ValueError(
                    "get_time_ordered_emission() got an argument for 'pixels', "
                    "but argument 'nside' is missing"
                )
            if lonlat:
                raise ValueError(
                    "get_time_ordered_emission() has 'lonlat' set to True "
                    "but 'theta' and 'phi' is not given"
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

        if (theta is not None) and (phi is not None):
            if theta.size != phi.size:
                raise ValueError(
                    "get_time_ordered_emission() got arguments 'theta' and 'phi' "
                    "with different size "
                )
            if theta.size == 1:
                phi = np.expand_dims(phi, axis=0)
                theta = np.expand_dims(theta, axis=0)
            if lonlat:
                theta = theta.to(u.deg)
                phi = phi.to(u.deg)
            else:
                theta = theta.to(u.rad)
                phi = phi.to(u.rad)

        if not isinstance(obs_time, Time):
            raise TypeError("argument 'obs_time' must be of type 'astropy.time.Time'")

        obs_pos, earth_pos = _get_solar_system_bodies(obs, obs_time, obs_pos)

        # If the specific value of freq is not covered by the fitted spectral
        # parameters in the model we interpolate/extrapolate to find a new
        # emissivity, albedo and scattering phase coefficients.
        freq = freq.to("GHz", equivalencies=u.spectral())
        model = self._model
        phase_coeffs = interp_phase_coeffs(
            freq,
            phase_coeffs=model.phase_coeffs,
            phase_coeffs_spectrum=model.phase_coeffs_spectrum,
        )
        spectral_comp_parameters = interp_comp_spectral_params(
            freq=freq,
            emissivities=model.emissivities,
            emissivity_spectrum=model.emissivity_spectrum,
            albedos=model.albedos,
            albedo_spectrum=model.albedo_spectrum,
        )

        if binned:
            if pixels is not None:
                unique_pixels, counts = np.unique(pixels, return_counts=True)
                unit_vectors = _get_ecliptic_unit_vectors(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
            else:
                unique_angles, counts = np.unique(
                    np.asarray([theta, phi]), return_counts=True, axis=1
                )
                unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
                unit_vectors = _get_ecliptic_unit_vectors(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )

            emission = np.zeros((model.ncomps, hp.nside2npix(nside)))
            for idx, (label, component) in enumerate(model.comps.items()):
                emission[idx, unique_pixels] = trapezoidal(
                    comp=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=obs_pos.value,
                    earth_pos=earth_pos.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=model.cloud_offset,
                    T_0=model.T_0.value,
                    delta=model.delta,
                    emissivity=spectral_comp_parameters[label]["emissivity"],
                    albedo=spectral_comp_parameters[label]["albedo"],
                    phase_coeffs=phase_coeffs,
                    colorcorr_table=colorcorr_table,
                )

            emission[:, unique_pixels] *= counts

        else:
            if pixels is not None:
                unique_pixels, indicies = np.unique(pixels, return_inverse=True)
                unit_vectors = _get_ecliptic_unit_vectors(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
                emission = np.zeros((model.ncomps, len(pixels)))

            else:
                unique_angles, indicies = np.unique(
                    np.asarray([theta, phi]), return_inverse=True, axis=1
                )
                unit_vectors = _get_ecliptic_unit_vectors(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )
                emission = np.zeros((model.ncomps, len(theta)))

            for idx, (label, component) in enumerate(model.comps.items()):
                integrated_comp_emission = trapezoidal(
                    comp=component,
                    freq=freq.value,
                    line_of_sight=self._line_of_sights[label].value,
                    observer_pos=obs_pos.value,
                    earth_pos=earth_pos.value,
                    unit_vectors=unit_vectors,
                    cloud_offset=model.cloud_offset,
                    T_0=model.T_0.value,
                    delta=model.delta,
                    emissivity=spectral_comp_parameters[label]["emissivity"],
                    albedo=spectral_comp_parameters[label]["albedo"],
                    phase_coeffs=phase_coeffs,
                    colorcorr_table=colorcorr_table,
                )

                emission[idx] = integrated_comp_emission[indicies]

        # The output unit is W/Hz/m^2/sr which we convert to MJy/sr
        emission = (emission << (u.W / u.Hz / u.m ** 2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)


def _get_solar_system_bodies(
    obs: str, obs_time: Time, obs_pos: Optional[Quantity]
) -> Tuple[Quantity[u.AU], Quantity[u.AU]]:
    """Returns the observer and Earth's position."""

    earth_skycoord = get_body("Earth", time=obs_time)
    earth_skycoord = earth_skycoord.transform_to(HeliocentricMeanEcliptic)
    earth_pos = earth_skycoord.represent_as("cartesian").xyz.to(u.AU)

    if obs_pos is None:
        if obs.lower() == "semb-l2":
            # We assume that L2 is located at a constant distance from the
            # Earth along Earths unit vector in heliocentric coordinates.
            earth_length = np.linalg.norm(earth_pos)
            earth_unit_vec = earth_pos / earth_length
            semb_l2_length = earth_length + DISTANCE_EARTH_TO_L2
            obs_pos_ = earth_unit_vec * semb_l2_length
        else:
            obs_skycoord = get_body(obs, time=obs_time)
            obs_skycoord = obs_skycoord.transform_to(HeliocentricMeanEcliptic)
            obs_pos_ = obs_skycoord.represent_as("cartesian").xyz.to(u.AU)
    else:
        obs_pos_ = obs_pos

    return obs_pos_.reshape(3, 1), earth_pos.reshape(3, 1)


def _get_ecliptic_unit_vectors(
    coord_in: str,
    pixels: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
    nside: Optional[int] = None,
    phi: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
    theta: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
    lonlat: bool = False,
) -> NDArray[np.floating]:
    """
    Since the Interplanetary Dust Model is evaluated in Ecliptic coordinates,
    we need to rotate any unit vectors (obtained from the pointing) defined in
    another coordinate frame to ecliptic before evaluating the model.
    """

    if pixels is None:
        unit_vectors = np.asarray(hp.ang2vec(theta, phi, lonlat=lonlat)).transpose()
    else:
        unit_vectors = np.asarray(hp.pix2vec(nside, pixels))

    return np.asarray(hp.Rotator(coord=[coord_in, "E"])(unit_vectors))
