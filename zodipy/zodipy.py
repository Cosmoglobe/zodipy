from __future__ import annotations

import functools
import multiprocessing
import platform
from typing import TYPE_CHECKING, Callable, Literal

import astropy_healpix as hp
import numpy as np
from astropy import coordinates as coords
from astropy import units
from scipy import interpolate

from zodipy._bandpass import get_bandpass_interpolation_table, validate_and_get_bandpass
from zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipy._coords import get_earth_skycoord, get_obs_skycoord, string_to_coordinate_frame
from zodipy._emission import EMISSION_MAPPING
from zodipy._interpolate_source import SOURCE_PARAMS_MAPPING
from zodipy._ipd_comps import ComponentLabel
from zodipy._ipd_dens_funcs import construct_density_partials_comps
from zodipy._line_of_sight import get_line_of_sight_range
from zodipy._validators import get_validated_ang
from zodipy.model_registry import model_registry

if TYPE_CHECKING:
    import numpy.typing as npt
    from astropy import time

PLATFORM = platform.system().lower()
SYS_PROC_START_METHOD = "fork" if "windows" not in PLATFORM else None


class Zodipy:
    """Main interface to ZodiPy.

    The zodiacal light simulations are configured by specifying a bandpass (`freq`, `weights`)
    or a delta/center frequency (`freq`), and a string representation of a built in interplanetary
    dust model (`model`). See https://cosmoglobe.github.io/zodipy/introduction/ for a list of
    available models.
    """

    def __init__(
        self,
        freq: units.Quantity,
        weights: npt.ArrayLike | None = None,
        model: str = "dirbe",
        gauss_quad_degree: int = 50,
        extrapolate: bool = False,
        interp_kind: str = "linear",
        ephemeris: str = "builtin",
        n_proc: int = 1,
    ) -> None:
        """Initialize the Zodipy interface.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            model (str): Name of the interplanetary dust model to use in the simulations.
                Defaults to DIRBE.
            gauss_quad_degree (int): Order of the Gaussian-Legendre quadrature used to evaluate
                the line-of-sight integral in the simulations. Default is 50 points.
            interp_kind (str): Interpolation kind used in `scipy.interpolate.interp1d` to
                interpolate spectral paramters (see [Scipy documentation](
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)).
                Defaults to 'linear'.
            extrapolate (bool): If `True` all spectral quantities in the selected model are
                extrapolated to the requested frequencies or wavelengths. If `False`, an
                exception is raised on requested frequencies/wavelengths outside of the
                valid model range. Default is `False`.
            ephemeris (str): Ephemeris used in `astropy.coordinates.solar_system_ephemeris` to
                compute the positions of the observer and the Earth. Defaults to 'builtin'. See the
                [Astropy documentation](https://docs.astropy.org/en/stable/coordinates/solarsystem.html)
                for available ephemerides.
            n_proc (int): Number of cores to use. If `n_proc` is greater than 1, the line-of-sight
                integrals are parallelized using the `multiprocessing` module. Defaults to 1.
        """
        self.ephemeris = ephemeris
        self.n_proc = n_proc

        self._interpolator = functools.partial(
            interpolate.interp1d,
            kind=interp_kind,
            fill_value="extrapolate" if extrapolate else np.nan,
        )
        self._ipd_model = model_registry.get_model(model)
        self._gauss_points_and_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)

        bandpass = validate_and_get_bandpass(
            freq=freq,
            weights=weights,
            model=self._ipd_model,
            extrapolate=extrapolate,
        )
        self._bandpass_interpolatation_table = get_bandpass_interpolation_table(bandpass)
        self._source_parameters = SOURCE_PARAMS_MAPPING[type(self._ipd_model)](
            bandpass, self._ipd_model, self._interpolator
        )

    def get_emission_skycoord(
        self,
        coord: coords.SkyCoord,
        *,
        obs_pos: units.Quantity | str = "earth",
        return_comps: bool = False,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal light for all observations in a `SkyCoord` object.

        Args:
            coord: `astropy.coordinates.SkyCoord` object representing the observations for which to
                simulate the zodiacal light. The `frame` and `obstime` attributes of the `SkyCoord`
                object must be set. The `obstime` attribute should correspond to a single
                observational time for which the zodiacal light is assumed to be stationary.
                Additionally, the frame must be convertible to
                `astropy.coordinates.BarycentricMeanEcliptic`.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            return_comps: If True, the emission is returned component-wise. Defaults to False.

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        (unique_lon, unique_lat), indicies = np.unique(
            np.vstack([coord.spherical.lon.value, coord.spherical.lat.value]),
            return_inverse=True,
            axis=1,
        )

        obs_time = coord.obstime
        if obs_time is None:
            msg = "The `obstime` attribute of the `SkyCoord` object must be set."
            raise ValueError(msg)

        coord = coords.SkyCoord(
            unique_lon,
            unique_lat,
            unit=units.deg,
            frame=coord.frame,
        )

        return self._compute_emission(
            obs_pos=obs_pos,
            obs_time=obs_time,
            coordinates=coord,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_emission_ang(
        self,
        theta: units.Quantity,
        phi: units.Quantity,
        *,
        lonlat: bool = False,
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        coord_in: Literal["E", "G", "C"] = "E",
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
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.
            return_comps: If True, the emission is returned component-wise. Defaults to False.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta=theta, phi=phi, lonlat=lonlat)

        (theta, phi), indicies = np.unique(np.stack([theta, phi]), return_inverse=True, axis=1)
        frame = string_to_coordinate_frame(coord_in)
        coordinates = coords.SkyCoord(theta, phi, frame=frame)

        return self._compute_emission(
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
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        coord_in: Literal["E", "G", "C"] = "E",
        return_comps: bool = False,
        order: Literal["ring", "nested"] = "ring",
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`.

        Args:
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            order: Order of the HEALPix grid.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        frame = string_to_coordinate_frame(coord_in)
        healpix = hp.HEALPix(nside=nside, order=order, frame=frame)
        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        coordinates = healpix.healpix_to_skycoord(unique_pixels)

        return self._compute_emission(
            obs_time=obs_time,
            obs_pos=obs_pos,
            coordinates=coordinates,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_binned_emission_skycoord(
        self,
        coord: coords.SkyCoord,
        *,
        nside: int,
        obs_pos: units.Quantity | str = "earth",
        return_comps: bool = False,
        order: Literal["ring", "nested"] = "ring",
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated binned zodiacal light for all observations in a `SkyCoord` object.

        Args:
            coord: `astropy.coordinates.SkyCoord` object representing the observations for which to
                simulate the zodiacal light. The `frame` and `obstime` attributes of the `SkyCoord`
                object must be set. The `obstime` attribute should correspond to a single
                observational time for which the zodiacal light is assumed to be stationary.
                Additionally, the frame must be convertible to
                `astropy.coordinates.BarycentricMeanEcliptic`.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            order: Order of the HEALPix grid.
            solar_cut: Angular distance around the Sun for which all pointing are discarded.
                Defaults to `None`.

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        (unique_lon, unique_lat), indicies = np.unique(
            np.vstack([coord.spherical.lon.value, coord.spherical.lat.value]),
            return_inverse=True,
            axis=1,
        )

        obs_time = coord.obstime
        if obs_time is None:
            msg = "The `obstime` attribute of the `SkyCoord` object must be set."
            raise ValueError(msg)

        coord = coords.SkyCoord(
            unique_lon,
            unique_lat,
            unit=units.deg,
            frame=coord.frame,
        )
        healpix = hp.HEALPix(nside, order=order, frame=coord.frame)
        return self._compute_emission(
            obs_pos=obs_pos,
            obs_time=obs_time,
            coordinates=coord,
            indicies=indicies,
            healpix=healpix,
            return_comps=return_comps,
            solar_cut=solar_cut,
        )

    def get_binned_emission_ang(
        self,
        theta: units.Quantity,
        phi: units.Quantity,
        *,
        lonlat: bool = False,
        nside: int,
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        coord_in: Literal["E", "G", "C"] = "E",
        return_comps: bool = False,
        order: Literal["ring", "nested"] = "ring",
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
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            order: Order of the HEALPix grid.
            solar_cut: Angular distance around the Sun for which all pointing are discarded.
                Defaults to `None`.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta, phi, lonlat=lonlat)
        frame = string_to_coordinate_frame(coord_in)
        healpix = hp.HEALPix(nside, order=order, frame=frame)
        (theta, phi), counts = np.unique(np.vstack([theta, phi]), return_counts=True, axis=1)
        coordinates = coords.SkyCoord(
            theta,
            phi,
            frame=frame,
        )

        return self._compute_emission(
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
        obs_time: time.Time,
        obs_pos: units.Quantity | str = "earth",
        coord_in: Literal["E", "G", "C"] = "E",
        return_comps: bool = False,
        order: Literal["ring", "nested"] = "ring",
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated binned zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`. The emission is binned to a HEALPix map with resolution given by
        `nside`.

        Args:
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation. This should be a single observational time.
            obs_pos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            order: Order of the HEALPix grid.
            solar_cut: Angular distance around the Sun for which all pointing are discarded.
                Defaults to `None`.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        frame = string_to_coordinate_frame(coord_in)
        healpix = hp.HEALPix(nside=nside, order=order, frame=frame)
        unique_pixels, counts = np.unique(pixels, return_counts=True)
        coordinates = healpix.healpix_to_skycoord(unique_pixels)

        return self._compute_emission(
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
        obs_time: time.Time,
        obs_pos: units.Quantity | str,
        coordinates: coords.SkyCoord,
        indicies: npt.NDArray,
        healpix: hp.HEALPix | None = None,
        return_comps: bool = False,
        solar_cut: units.Quantity | None = None,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Compute the zodiacal light for a given configuration."""
        earth_skycoord = get_earth_skycoord(obs_time, ephemeris=self.ephemeris)
        obs_skycoord = get_obs_skycoord(obs_pos, obs_time, earth_skycoord, ephemeris=self.ephemeris)

        coordinates = coordinates.transform_to(coords.BarycentricMeanEcliptic)

        bin_output_to_healpix_map = healpix is not None
        filter_coords_by_solar_cut = solar_cut is not None
        distribute_to_cores = self.n_proc > 1 and coordinates.size > self.n_proc

        if bin_output_to_healpix_map and filter_coords_by_solar_cut:
            sun_skycoord = coords.SkyCoord(
                obs_skycoord.spherical.lon + 180 * units.deg,
                obs_skycoord.spherical.lat,
                frame=coords.BarycentricMeanEcliptic,
            )
            angular_distance = coordinates.separation(sun_skycoord)
            solar_mask = angular_distance < solar_cut
            coordinates = coordinates[~solar_mask]

        unit_vectors = coordinates.cartesian.xyz.value

        start, stop = get_line_of_sight_range(
            components=self._ipd_model.comps.keys(),
            unit_vectors=unit_vectors,
            obs_pos=obs_skycoord.cartesian.xyz.to_value(units.AU),
        )

        density_partials = construct_density_partials_comps(
            comps=self._ipd_model.comps,
            dynamic_params={
                "X_earth": earth_skycoord.cartesian.xyz.to_value(units.AU)[
                    :, np.newaxis, np.newaxis
                ]
            },
        )

        common_integrand = functools.partial(
            EMISSION_MAPPING[type(self._ipd_model)],
            X_obs=obs_skycoord.cartesian.xyz.to_value(units.AU)[:, np.newaxis, np.newaxis],
            bp_interpolation_table=self._bandpass_interpolatation_table,
            **self._source_parameters["common"],
        )

        if distribute_to_cores:
            unit_vector_chunks = np.array_split(unit_vectors, self.n_proc, axis=-1)
            integrated_comp_emission = np.zeros((self._ipd_model.ncomps, coordinates.size))

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
                            **self._source_parameters[comp_label],
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
            integrated_comp_emission = np.zeros((self._ipd_model.ncomps, coordinates.size))
            unit_vectors_expanded = np.expand_dims(unit_vectors, axis=-1)

            for idx, comp_label in enumerate(self._ipd_model.comps.keys()):
                comp_integrand = functools.partial(
                    common_integrand,
                    u_los=unit_vectors_expanded,
                    start=np.expand_dims(start[comp_label], axis=-1),
                    stop=np.expand_dims(stop[comp_label], axis=-1),
                    get_density_function=density_partials[comp_label],
                    **self._source_parameters[comp_label],
                )

                integrated_comp_emission[idx] = (
                    _integrate_gauss_quad(comp_integrand, *self._gauss_points_and_weights)
                    * 0.5
                    * (stop[comp_label] - start[comp_label])
                )

        if bin_output_to_healpix_map:
            emission = np.zeros((self._ipd_model.ncomps, healpix.npix))  # type: ignore
            pixels = healpix.skycoord_to_healpix(coordinates)  # type: ignore
            emission[:, pixels] = integrated_comp_emission
        else:
            emission = np.zeros((self._ipd_model.ncomps, indicies.size))
            emission = integrated_comp_emission[:, indicies]

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(units.MJy / units.sr)

        return emission if return_comps else emission.sum(axis=0)

    @property
    def supported_observers(self) -> list[str]:
        """Return a list of available observers given an ephemeris."""
        with coords.solar_system_ephemeris.set(self.ephemeris):
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
