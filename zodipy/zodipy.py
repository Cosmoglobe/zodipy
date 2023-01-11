from __future__ import annotations

import multiprocessing
import platform
from functools import partial
from typing import Callable, Literal, Sequence

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.typing as npt
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time

from zodipy._bandpass import get_bandpass_interpolation_table, validate_and_get_bandpass
from zodipy._constants import SPECIFIC_INTENSITY_UNITS
from zodipy._emission import EMISSION_MAPPING
from zodipy._interpolate_source import SOURCE_PARAMS_MAPPING
from zodipy._ipd_comps import ComponentLabel
from zodipy._ipd_dens_funcs import construct_density_partials_comps
from zodipy._line_of_sight import get_line_of_sight_start_and_stop_distances
from zodipy._sky_coords import get_obs_and_earth_positions
from zodipy._types import FrequencyOrWavelength, Pixels, SkyAngles
from zodipy._unit_vectors import get_unit_vectors_from_ang, get_unit_vectors_from_pixels
from zodipy._validators import get_validated_ang, get_validated_pix
from zodipy.model_registry import model_registry

PLATFORM = platform.system().lower()
SYS_PROC_START_METHOD = "fork" if "windows" not in PLATFORM else None

ParameterDict = dict


class Zodipy:
    """Interface for simulating zodiacal emission.

    Sets up the simulation configuration and provides methods for simulating the zodiacal
    emission that a solar system observer is predicted to see given an interplanetary dust
    model.

    Attributes:
        model (str): Name of the interplanetary dust model to use in the simulations.
            Defaults to DIRBE.
        extrapolate (bool): If `True` all spectral quantities in the selected model are
            linearly extrapolated to the requested frequency/wavelength. If `False`, an
            exception is raised on requested frequencies/wavelengths outside of the
            valid model range. Default is `False`.
        parallel (bool): If `True`, input pointing sequences will be split among all
            available cores on the machine, and the emission will be computed in parallel.
            This is useful for large pointing chunks. Default is `False`.
        n_proc (int): Number of cores to use when parallel computation. Defaults is None,
            which will use all available cores.
        solar_cut (u.Quantity[u.deg]): Cutoff angle from the sun in degrees. The emission
            for all the pointing with angular distance between the sun smaller than
            `solar_cutoff` are set to `np.nan`. This is due to the model singularity of the
            diffuse cloud component at the heliocentric origin. Such a cutoff may be
            useful when using simulated pointing, but actual scanning strategies are
            unlikely to look directly at the sun. This feature is turned of by setting this
            argument to `None`. Defaults to 5 degrees.
        solar_cut_fill_value (float): Fill value for the masked solar cut pointing.
            Defaults to `np.nan`.
        gauss_quad_degree (int): Order of the Gaussian-Legendre quadrature used to evaluate
            the brightness integral. Default is 50 points.
        ephemeris (str): Ephemeris used to compute the positions of the observer and the
            Earth. Defaults to 'de432s' which requires downloading (and caching) a ~10MB
            file. For more information on available ephemeridis, please visit
            https://docs.astropy.org/en/stable/coordinates/solarsystem.html

    """

    def __init__(
        self,
        model: str = "dirbe",
        extrapolate: bool = False,
        parallel: bool = False,
        n_proc: int | None = None,
        solar_cut: u.Quantity[u.deg] | None = None,
        solar_cut_fill_value: float = np.nan,
        gauss_quad_degree: int = 50,
        ephemeris: str = "de432s",
    ) -> None:

        self.model = model
        self.extrapolate = extrapolate
        self.parallel = parallel
        self.n_proc = n_proc
        self.solar_cut = solar_cut.to(u.rad) if solar_cut is not None else solar_cut
        self.solar_cut_fill_value = solar_cut_fill_value
        self.gauss_quad_degree = gauss_quad_degree
        self.ephemeris = ephemeris

        self.ipd_model = model_registry.get_model(model)
        self.gauss_points_and_weights = np.polynomial.legendre.leggauss(
            gauss_quad_degree
        )

    @property
    def supported_observers(self) -> list[str]:
        """Returns a list of available observers given an ephemeris."""
        return list(solar_system_ephemeris.bodies) + ["semb-l2"]

    def get_parameters(self) -> ParameterDict:
        """Returns a dictionary containing the interplanetary dust model parameters."""
        return self.ipd_model.to_dict()

    def update_parameters(self, parameters: ParameterDict) -> None:
        """Updates the interplanetary dust model parameters.

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
                        self.ipd_model.comps[ComponentLabel(comp_key)]
                    )(**comp_value)
            elif isinstance(value, dict):
                _dict[key] = {ComponentLabel(k): v for k, v in value.items()}

        self.ipd_model = self.ipd_model.__class__(**_dict)

    def get_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: Sequence[float] | npt.NDArray[np.floating] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be either
                radians or degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be either
                radians or degrees.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta=theta, phi=phi, lonlat=lonlat)

        unique_angles, indicies = np.unique(
            np.asarray([theta, phi]), return_inverse=True, axis=1
        )
        unit_vectors = get_unit_vectors_from_ang(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: Pixels,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: Sequence[float] | npt.NDArray[np.floating] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        pixels = get_validated_pix(pixels=pixels, nside=nside)

        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def get_binned_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: Sequence[float] | npt.NDArray[np.floating] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`. The emission is binned to a HEALPix map with
        resolution given by `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be either
                radians or degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be either
                radians or degrees.
            nside: HEALPix resolution parameter of the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            lonlat: If True, input angles `theta`, `phi` are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        theta, phi = get_validated_ang(theta=theta, phi=phi, lonlat=lonlat)

        unique_angles, counts = np.unique(
            np.asarray([theta, phi]), return_counts=True, axis=1
        )
        unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
        unit_vectors = get_unit_vectors_from_ang(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def get_binned_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: Pixels,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: Sequence[float] | npt.NDArray[np.floating] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`. The emission is binned to a HEALPix map with resolution given by
        `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission. The frequencies
                must be strictly increasing.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights
                are assumed to be given in spectral radiance units (Jy/sr).
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """
        pixels = get_validated_pix(pixels=pixels, nside=nside)

        unique_pixels, counts = np.unique(pixels, return_counts=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def _compute_emission(
        self,
        freq: FrequencyOrWavelength,
        weights: Sequence[float] | npt.NDArray[np.floating] | None,
        obs: str,
        obs_time: Time,
        unit_vectors: npt.NDArray[np.float64],
        indicies: npt.NDArray[np.int64],
        binned: bool = False,
        obs_pos: u.Quantity[u.AU] | None = None,
        pixels: npt.NDArray[np.int64] | None = None,
        nside: int | None = None,
        return_comps: bool = False,
    ) -> u.Quantity[u.MJy / u.sr]:
        """Computes the component-wise zodiacal emission."""
        bandpass = validate_and_get_bandpass(
            freq=freq,
            weights=weights,
            model=self.ipd_model,
            extrapolate=self.extrapolate,
        )

        # Get model parameters, some of which have been interpolated to the given
        # frequency or bandpass.
        source_parameters = SOURCE_PARAMS_MAPPING[type(self.ipd_model)](
            bandpass, self.ipd_model
        )

        observer_position, earth_position = get_obs_and_earth_positions(
            obs=obs, obs_time=obs_time, obs_pos=obs_pos
        )

        # Get the integration limits for each zodiacal component (which may be
        # different or the same depending on the model) along all line of sights.
        start, stop = get_line_of_sight_start_and_stop_distances(
            components=self.ipd_model.comps.keys(),
            unit_vectors=unit_vectors,
            obs_pos=observer_position,
        )

        density_partials = construct_density_partials_comps(
            comps=self.ipd_model.comps,
            dynamic_params={"X_earth": earth_position},
        )

        # Make table of pre-computed bandpass integrated blackbody emission.
        bandpass_interpolatation_table = get_bandpass_interpolation_table(bandpass)

        common_integrand = partial(
            EMISSION_MAPPING[type(self.ipd_model)],
            X_obs=observer_position,
            bp_interpolation_table=bandpass_interpolatation_table,
            **source_parameters["common"],
        )

        if self.parallel:
            # Distribute pointing to cores.
            n_proc = multiprocessing.cpu_count() if self.n_proc is None else self.n_proc

            unit_vector_chunks = np.array_split(unit_vectors, n_proc, axis=-1)
            integrated_comp_emission = np.zeros(
                (len(self.ipd_model.comps), unit_vectors.shape[1])
            )
            with multiprocessing.get_context(SYS_PROC_START_METHOD).Pool(
                processes=n_proc
            ) as pool:
                # Create a partial functions that will be passed to the quadpy integration
                # scheme. Arrays are reshaped to (d, n, p) for broadcasting purposes where
                # d is the geometrical dimensionality, n is the number of different
                # pointings, and p is the number of integration points of the quadrature.

                for idx, comp_label in enumerate(self.ipd_model.comps.keys()):
                    stop_chunks = np.array_split(stop[comp_label], n_proc, axis=-1)
                    if start[comp_label].size == 1:
                        start_chunks = [start[comp_label]] * n_proc
                    else:
                        start_chunks = np.array_split(
                            start[comp_label], n_proc, axis=-1
                        )
                    comp_integrands = [
                        partial(
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
                            args=(comp_integrand, self.gauss_points_and_weights),
                        )
                        for comp_integrand in comp_integrands
                    ]

                    integrated_comp_emission[idx] += (
                        np.concatenate([result.get() for result in proc_chunks])
                        * 0.5
                        * (stop[comp_label] - start[comp_label])
                    )

        else:
            integrated_comp_emission = np.zeros(
                (len(self.ipd_model.comps), unit_vectors.shape[1])
            )
            unit_vectors_expanded = np.expand_dims(unit_vectors, axis=-1)

            for idx, comp_label in enumerate(self.ipd_model.comps.keys()):
                comp_integrand = partial(
                    common_integrand,
                    u_los=unit_vectors_expanded,
                    start=np.expand_dims(start[comp_label], axis=-1),
                    stop=np.expand_dims(stop[comp_label], axis=-1),
                    get_density_function=density_partials[comp_label],
                    **source_parameters[comp_label],
                )

                integrated_comp_emission[idx] = (
                    _integrate_gauss_quad(comp_integrand, self.gauss_points_and_weights)
                    * 0.5
                    * (stop[comp_label] - start[comp_label])
                )

        emission = np.zeros(
            (
                len(self.ipd_model.comps),
                hp.nside2npix(nside) if binned else indicies.size,
            )
        )
        if binned:
            emission[:, pixels] = integrated_comp_emission
        else:
            emission = integrated_comp_emission[:, indicies]

        if self.solar_cut is not None:
            # The observer position is aquired in geocentric coordinates before being
            # rotated to ecliptic coordinates. This means that we can find the
            # heliocentric origin in this coordinate system by simply taking the negative
            # of the observer position. Emission from unit_vectors with an angular distance
            # (between the pointing and the sun) smaller than the specified `solar_cutoff`
            # value is masked with the `solar_cutoff_fill_value`.
            ang_dist = hp.rotator.angdist(-observer_position.flatten(), unit_vectors)
            solar_mask = ang_dist < self.solar_cut.value
            if binned and pixels is not None:
                emission[:, pixels[solar_mask]] = self.solar_cut_fill_value
            else:
                emission[:, solar_mask[indicies]] = self.solar_cut_fill_value

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

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
    points_and_weights: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Integrate the emission from a component using Gauss-Legendre quadrature."""
    return np.squeeze(sum(fn(x) * w for x, w in zip(*points_and_weights)))
