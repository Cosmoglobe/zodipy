from __future__ import annotations

import functools
import multiprocessing
import platform
from typing import TYPE_CHECKING, Callable

import astropy_healpix as hp
import numpy as np
from astropy import coordinates as coords
from astropy import units
from scipy import interpolate

from zodipy._bandpass import get_bandpass_interpolation_table, validate_and_get_bandpass
from zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipy._coords import get_earth_skycoord, get_obs_skycoord
from zodipy._emission import EMISSION_MAPPING
from zodipy._interpolate_source import SOURCE_PARAMS_MAPPING
from zodipy._ipd_comps import ComponentLabel
from zodipy._ipd_dens_funcs import construct_density_partials_comps
from zodipy._line_of_sight import get_line_of_sight_start_and_stop_distances
from zodipy._validators import get_validated_ang
from zodipy.model_registry import model_registry

if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy import time

PLATFORM = platform.system().lower()
SYS_PROC_START_METHOD = "fork" if "windows" not in PLATFORM else None


class Zodipy:
    """Interface for simulating zodiacal emission.

    This class provides methods for simulating zodiacal emission given observer pointing
    either in sky angles or through HEALPix pixels.

    Args:
        model (str): Name of the interplanetary dust model to use in the simulations.
            Defaults to DIRBE.
        gauss_quad_degree (int): Order of the Gaussian-Legendre quadrature used to evaluate
            the line-of-sight integral in the simulations. Default is 50 points.
        interp_kind (str): Interpolation kind used to interpolate relevant model parameters.
            Defaults to 'linear'. For more information on available interpolation methods,
            please visit the [Scipy documentation](
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
        extrapolate (bool): If `True` all spectral quantities in the selected model are
            extrapolated to the requested frequencies or wavelengths. If `False`, an
            exception is raised on requested frequencies/wavelengths outside of the
            valid model range. Default is `False`.
        ephemeris (str): Ephemeris used to compute the positions of the observer and the
            Earth. Defaults to 'de432s', which requires downloading (and caching) a ~10MB
            file. For more information on available ephemeridis, please visit the [Astropy
            documentation](https://docs.astropy.org/en/stable/coordinates/solarsystem.html)
        n_proc (int): Number of cores to use. If `n_proc` is greater than 1, the line-of-sight
            integrals are parallelized using the `multiprocessing` module. Defaults to 1.

    """

    def __init__(
        self,
        model: str = "dirbe",
        gauss_quad_degree: int = 50,
        extrapolate: bool = False,
        interp_kind: str = "linear",
        ephemeris: str = "de432s",
        n_proc: int = 1,
    ) -> None:
        self.model = model
        self.gauss_quad_degree = gauss_quad_degree
        self.extrapolate = extrapolate
        self.interp_kind = interp_kind
        self.ephemeris = ephemeris
        self.n_proc = n_proc

        self._interpolator = functools.partial(
            interpolate.interp1d,
            kind=self.interp_kind,
            fill_value="extrapolate" if self.extrapolate else np.nan,
        )
        self._ipd_model = model_registry.get_model(model)
        self._gauss_points_and_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)

    @property
    def supported_observers(self) -> list[str]:
        """Return a list of available observers given an ephemeris."""
        return [*list(coords.solar_system_ephemeris.bodies), "semb-l2"]

    def get_parameters(self) -> dict:
        """Return a dictionary containing the interplanetary dust model parameters."""
        return self._ipd_model.to_dict()

    def update_parameters(self, parameters: dict) -> None:
        """Update the interplanetary dust model parameters.

        Args:
            parameters: Dictionary of parameters to update. The keys must be the names
                of the parameters as defined in the model. To get the parameters dict
                of an existing model, use `Zodipy("dirbe").get_parameters()`.

        """
        _dict = parameters.copy()
        _dict["comps"] = {}
        for key, value in parameters.items():
            if key == "comps":
                for comp_key, comp_value in value.items():
                    _dict["comps"][ComponentLabel(comp_key)] = type(
                        self._ipd_model.comps[ComponentLabel(comp_key)]
                    )(**comp_value)
            elif isinstance(value, dict):
                _dict[key] = {ComponentLabel(k): v for k, v in value.items()}

        self._ipd_model = self._ipd_model.__class__(**_dict)

    def get_emission_skycoord(
        self,
        coord: coords.SkyCoord,
        *,
        obs_time: time.Time,
        freq: units.Quantity,
        obs_pos: units.Quantity | str = "earth",
        weights: npt.ArrayLike | None = None,
        return_comps: bool = False,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal light for observations in an Astropy `SkyCoord` object.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`.

        Args:
            coord: Astropy `SkyCoord` object representing the pointing on the sky. This
                object must have a frame attribute representing the coordinate frame of the
                input pointing, for example `astropy.coordinates.Galactic`. The frame must
                be convertible to `BarycentricMeanEcliptic`.
            obs_time: Time of observation. This should be a single observational time.
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            obs_pos: The heliocentric ecliptic position of the observer in AU, or a string
                representing an observer in the `astropy.coordinates.solar_system_ephemeris`.
                This should correspond to a single position. Defaults to 'earth'.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            return_comps: If True, the emission is returned component-wise. Defaults to False.

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        (unique_lon, unique_lat), indicies = np.unique(
            np.vstack([coord.spherical.lon.value, coord.spherical.lat.value]),
            return_inverse=True,
            axis=1,
        )
        coord = coords.SkyCoord(unique_lon * units.deg, unique_lat * units.deg, frame=coord.frame)

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coord,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_emission_ang(
        self,
        theta: units.Quantity,
        phi: units.Quantity,
        *,
        freq: units.Quantity,
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        frame: type[coords.BaseCoordinateFrame] = coords.BarycentricMeanEcliptic,
        weights: npt.ArrayLike | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`, matching the healpy convention.

        Args:
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be compatible
                with degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be compatible
                with degrees.
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer in AU, or a string
                representing an observer in the `astropy.coordinates.solar_system_ephemeris`.
                This should correspond to a single position. Defaults to 'earth'.
            frame: Astropy coordinate frame representing the coordinate frame of the input pointing.
                Default is `BarycentricMeanEcliptic`, corresponding to ecliptic coordinates. Other
                alternatives are `Galactic` and `ICRS`.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta=theta, phi=phi, lonlat=lonlat)

        (theta, phi), indicies = np.unique(np.stack([theta, phi]), return_inverse=True, axis=1)
        coordinates = coords.SkyCoord(
            theta,
            phi,
            frame=frame,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coordinates,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_emission_pix(
        self,
        pixels: npt.ArrayLike,
        *,
        nside: int,
        freq: units.Quantity,
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        frame: type[coords.BaseCoordinateFrame] = coords.BarycentricMeanEcliptic,
        weights: npt.ArrayLike | None = None,
        return_comps: bool = False,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`.

        Args:
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer in AU, or a string
                representing an observer in the `astropy.coordinates.solar_system_ephemeris`.
                This should correspond to a single position. Defaults to 'earth'.
            frame: Astropy coordinate frame representing the coordinate frame of the input pointing.
                Default is `BarycentricMeanEcliptic`, corresponding to ecliptic coordinates. Other
                alternatives are `Galactic` and `ICRS`.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            return_comps: If True, the emission is returned component-wise. Defaults to False.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        healpix = hp.HEALPix(nside=nside, order="ring", frame=frame)
        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        coordinates = healpix.healpix_to_skycoord(unique_pixels)

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coordinates,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_binned_emission_ang(
        self,
        theta: units.Quantity,
        phi: units.Quantity,
        *,
        nside: int,
        freq: units.Quantity,
        obs_time: time.Time,
        obs_pos: units.Quantity[units.AU] | str = "earth",
        frame: type[coords.BaseCoordinateFrame] = coords.BarycentricMeanEcliptic,
        weights: npt.ArrayLike | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated binned zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`, matching the healpy convention. The emission is
        binned to a HEALPix map with resolution given by `nside`.

        Args:
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be either
                radians or degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be either
                radians or degrees.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer in AU, or a string
                representing an observer in the `astropy.coordinates.solar_system_ephemeris`.
                This should correspond to a single position. Defaults to 'earth'.
            frame: Astropy coordinate frame representing the coordinate frame of the input pointing.
                Default is `BarycentricMeanEcliptic`, corresponding to ecliptic coordinates. Other
                alternatives are `Galactic` and `ICRS`.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            solar_cut: Cutoff angle from the sun. The emission for all the pointing with angular
                distance between the sun smaller than `solar_cut` are masked. Defaults to `None`.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta=theta, phi=phi, lonlat=lonlat)
        healpix = hp.HEALPix(nside, order="ring", frame=frame)
        (theta, phi), counts = np.unique(np.vstack([theta, phi]), return_counts=True, axis=1)
        coordinates = coords.SkyCoord(
            theta,
            phi,
            frame=frame,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coordinates,
            indicies=counts,
            healpix=healpix,
            return_comps=return_comps,
            solar_cut=solar_cut,
        )

    def get_binned_emission_pix(
        self,
        pixels: npt.ArrayLike,
        *,
        nside: int,
        freq: units.Quantity,
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        frame: type[coords.BaseCoordinateFrame] | str = coords.BarycentricMeanEcliptic,
        weights: npt.ArrayLike | None = None,
        return_comps: bool = False,
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated binned zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`. The emission is binned to a HEALPix map with resolution given by
        `nside`.

        Args:
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer in AU, or a string
                representing an observer in the `astropy.coordinates.solar_system_ephemeris`.
                This should correspond to a single position. Defaults to 'earth'.
            frame: Astropy coordinate frame representing the coordinate frame of the input pointing.
                Default is `BarycentricMeanEcliptic`, corresponding to ecliptic coordinates. Other
                alternatives are `Galactic` and `ICRS`.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            solar_cut: Cutoff angle from the sun. The emission for all the pointing with angular
                distance between the sun smaller than `solar_cut` are masked. Defaults to `None`.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        healpix = hp.HEALPix(nside=nside, order="ring", frame=frame)
        unique_pixels, counts = np.unique(pixels, return_counts=True)
        coordinates = healpix.healpix_to_skycoord(unique_pixels)

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coordinates,
            indicies=counts,
            healpix=healpix,
            return_comps=return_comps,
            solar_cut=solar_cut,
        )

    def _compute_emission(
        self,
        freq: units.Quantity,
        weights: npt.ArrayLike | None,
        obs_time: time.Time,
        obs_pos: units.Quantity | str,
        coordinates: coords.SkyCoord,
        indicies: npt.NDArray,
        healpix: hp.HEALPix | None = None,
        return_comps: bool = False,
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Compute the zodiacal light for a given configuration."""
        bandpass = validate_and_get_bandpass(
            freq=freq,
            weights=weights,
            model=self._ipd_model,
            extrapolate=self.extrapolate,
        )

        # Get model parameters, some of which have been interpolated to the given
        # frequency or bandpass.
        source_parameters = SOURCE_PARAMS_MAPPING[type(self._ipd_model)](
            bandpass, self._ipd_model, self._interpolator
        )

        earth_skycoord = get_earth_skycoord(obs_time)
        obs_skycoord = get_obs_skycoord(obs_pos, obs_time, earth_skycoord)

        # Rotate to ecliptic coordinates to evaluate zodiacal light model
        coordinates = coordinates.transform_to(coords.BarycentricMeanEcliptic)
        unit_vectors = coordinates.cartesian.xyz.value

        # Get the integration limits for each zodiacal component (which may be
        # different or the same depending on the model) along all line of sights.
        start, stop = get_line_of_sight_start_and_stop_distances(
            components=self._ipd_model.comps.keys(),
            unit_vectors=unit_vectors,
            obs_pos=obs_skycoord.cartesian.xyz.value,
        )

        density_partials = construct_density_partials_comps(
            comps=self._ipd_model.comps,
            dynamic_params={
                "X_earth": earth_skycoord.cartesian.xyz.value[:, np.newaxis, np.newaxis]
            },
        )

        bandpass_interpolatation_table = get_bandpass_interpolation_table(bandpass)

        common_integrand = functools.partial(
            EMISSION_MAPPING[type(self._ipd_model)],
            X_obs=obs_skycoord.cartesian.xyz.value[:, np.newaxis, np.newaxis],
            bp_interpolation_table=bandpass_interpolatation_table,
            **source_parameters["common"],
        )

        # Parallelize the line-of-sight integrals if more than one processor is used and the
        # number of unique observations is greater than the number of processors.
        if self.n_proc > 1 and unit_vectors.shape[-1] > self.n_proc:
            unit_vector_chunks = np.array_split(unit_vectors, self.n_proc, axis=-1)
            integrated_comp_emission = np.zeros((len(self._ipd_model.comps), unit_vectors.shape[1]))
            with multiprocessing.get_context(SYS_PROC_START_METHOD).Pool(
                processes=self.n_proc
            ) as pool:
                for idx, comp_label in enumerate(self._ipd_model.comps.keys()):
                    stop_chunks = np.array_split(stop[comp_label], self.n_proc, axis=-1)
                    if start[comp_label].size == 1:
                        start_chunks = [start[comp_label]] * self.n_proc
                    else:
                        start_chunks = np.array_split(start[comp_label], self.n_proc, axis=-1)
                    comp_integrands = [
                        functools.partial(
                            common_integrand,
                            u_los=np.expand_dims(unit_vectors, axis=-1),
                            start=np.expand_dims(start, axis=-1),
                            stop=np.expand_dims(stop, axis=-1),
                            get_density_function=density_partials[comp_label],
                            **source_parameters[comp_label],
                        )
                        for unit_vectors, start, stop in zip(
                            unit_vector_chunks, start_chunks, stop_chunks
                        )
                    ]

                    proc_chunks = [
                        pool.apply_async(
                            _integrate_gauss_quad,
                            args=(comp_integrand, *self._gauss_points_and_weights),
                        )
                        for comp_integrand in comp_integrands
                    ]

                    integrated_comp_emission[idx] += (
                        np.concatenate([result.get() for result in proc_chunks])
                        * 0.5
                        * (stop[comp_label] - start[comp_label])
                    )

        else:
            integrated_comp_emission = np.zeros((len(self._ipd_model.comps), unit_vectors.shape[1]))
            unit_vectors_expanded = np.expand_dims(unit_vectors, axis=-1)

            for idx, comp_label in enumerate(self._ipd_model.comps.keys()):
                comp_integrand = functools.partial(
                    common_integrand,
                    u_los=unit_vectors_expanded,
                    start=np.expand_dims(start[comp_label], axis=-1),
                    stop=np.expand_dims(stop[comp_label], axis=-1),
                    get_density_function=density_partials[comp_label],
                    **source_parameters[comp_label],
                )

                integrated_comp_emission[idx] = (
                    _integrate_gauss_quad(comp_integrand, *self._gauss_points_and_weights)
                    * 0.5
                    * (stop[comp_label] - start[comp_label])
                )

        # Output is requested to be binned
        if healpix:
            emission = np.zeros((len(self._ipd_model.comps), healpix.npix))
            pixels = healpix.skycoord_to_healpix(coordinates)
            emission[:, pixels] = integrated_comp_emission
            if solar_cut is not None:
                sun_skycoord = coords.SkyCoord(
                    obs_skycoord.spherical.lon + 180 * units.deg,
                    obs_skycoord.spherical.lat,
                    frame=coords.BarycentricMeanEcliptic,
                )
                angular_distance = coordinates.separation(sun_skycoord)
                solar_mask = angular_distance < solar_cut
                emission[:, pixels[solar_mask]] = np.nan

        else:
            emission = np.zeros((len(self._ipd_model.comps), indicies.size))
            emission = integrated_comp_emission[:, indicies]

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(units.MJy / units.sr)

        return emission if return_comps else emission.sum(axis=0)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for attribute_name, attribute in self.__dict__.items():
            if attribute_name.startswith("_"):
                continue
            repr_str += f"{attribute_name}={attribute!r}, "

        return repr_str[:-2] + ")"


def _integrate_gauss_quad(
    fn: Callable[[float], npt.NDArray[np.float64]],
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Integrate a function using Gauss-Legendre quadrature."""
    return np.squeeze(sum(fn(x) * w for x, w in zip(points, weights)))
