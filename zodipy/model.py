from __future__ import annotations

import functools
import multiprocessing
import platform
import typing

import numpy as np
import numpy.typing as npt
from astropy import coordinates as coords
from astropy import time, units
from scipy import integrate

from zodipy.blackbody import tabulate_blackbody_emission
from zodipy.bodies import get_earthpos_xyz, get_obspos_xyz
from zodipy.component import ComponentLabel
from zodipy.line_of_sight import (
    get_line_of_sight_range_dicts,
    integrate_leggauss,
)
from zodipy.model_registry import model_registry
from zodipy.number_density import get_partial_number_density_func, update_partial_earth_pos
from zodipy.unpack_model import get_model_interp_func

PLATFORM_METHOD = "fork" if "windows" not in platform.system().lower() else None


class Model:
    """Main interface to ZodiPy."""

    def __init__(
        self,
        x: units.Quantity,
        *,
        weights: npt.ArrayLike | None = None,
        name: str = "dirbe",
        gauss_quad_degree: int = 50,
        extrapolate: bool = False,
        ephemeris: str = "builtin",
    ) -> None:
        """Initialize the ZodiPy model interface.

        Args:
            x: Wavelength or frequency. If `x` is a sequence, it is assumed to be a the points
                corresponding to a bandpass and the corresponding `weights` must be provided.
            weights: Bandpass weights corresponding the the frequencies or wavelengths in `x`. The
                weights are assumed to represent a normalized instrument response in units of
                spectral radiance (Jy/sr).
            name: Zodiacal light model to use for the simulations. For a list of available models,
                see https://cosmoglobe.github.io/zodipy/introduction/. Defaults to 'dirbe'.
            gauss_quad_degree: Order of the Gaussian-legendre quadrature used to evaluate the
                line-of-sight integrals in the simulations. Default is 50 points.
            extrapolate: If `True` all spectral quantities in the selected model are extrapolated to
                the requested frequencies or wavelengths. If `False`, an exception is raised on
                requested values of `x` outside of the valid model range. Default is `False`.
            ephemeris: Ephemeris used in `astropy.coordinates.solar_system_ephemeris` to compute the
                positions of the observer and Earth. Defaults to 'builtin'. See the
                [Astropy documentation](https://docs.astropy.org/en/stable/coordinates/solarsystem.html)
                for available ephemerides.

        """
        try:
            if not x.isscalar and weights is None:
                msg = "Bandpass weights must be provided for non-scalar `x`."
                raise ValueError(msg)
        except AttributeError as error:
            msg = "The input 'x' must be an astropy Quantity."
            raise TypeError(msg) from error
        if x.isscalar and weights is not None:
            msg = "Bandpass weights should not be provided for scalar `x`."
            raise ValueError(msg)

        self._ipd_model = model_registry.get_model(name)

        if not extrapolate and not self._ipd_model.is_valid_at(x):
            msg = (
                "The requested frequencies are outside the valid range of the model. "
                "If this was intended, set the extrapolate argument to True."
            )
            raise ValueError(msg)

        # Bandpass is provided rather than a delta wavelength or frequency.
        if weights is not None:
            weights = np.asarray(weights)
            if x.size != weights.size:
                msg = "Number of wavelengths and weights must be the same in the bandpass."
                raise ValueError(msg)
            normalized_weights = weights / integrate.trapezoid(weights, x)
        else:
            normalized_weights = None

        self._x = x
        self._normalized_weights = normalized_weights
        self._b_nu_table = tabulate_blackbody_emission(self._x, self._normalized_weights)

        quad_points, quad_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)
        self._integrate_leggauss = functools.partial(
            integrate_leggauss,
            points=quad_points,
            weights=quad_weights,
        )

        self._ephemeris = ephemeris

        # Make mypy happy by declaring types of to-be initialized attributes.
        self._number_density_partials: dict[ComponentLabel, functools.partial]
        self._interped_comp_params: dict[ComponentLabel, dict]
        self._interped_shared_params: dict

        self._init_ipd_model_partials()

    def evaluate(
        self,
        skycoord: coords.SkyCoord,
        *,
        obspos: units.Quantity | str = "earth",
        return_comps: bool = False,
        nprocesses: int = 1,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal light.

        The zodiacal light is simulated for a single, or a sequence of observations. If a single
        `obspos` and `obstime` is provided for multiple coordinates, all coordinates are assumed to
        be observed from that position at that time. Otherwise, each coordinate is simulated from
        the corresponding observer position and time.

        Args:
            skycoord: `astropy.coordinates.SkyCoord` object representing the coordinates or
                observations for which to simulate the zodiacal light. The `frame` and `obstime`
                attributes of the `SkyCoord` object must be set. The `obstime` attribute must be
                specified, and correspond to a single, or a sequence of observational times with
                length matching the number of coordinates. The frame must be convertible to the
                `astropy.coordinates.BarycentricMeanEcliptic` frame.
            obspos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. If an explicit
                position is given, it must either be a single, or a sequence of positions with
                shape matching the number of coordinates Defaults to 'earth'.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            nprocesses: Number of cores to use. If `nprocesses >= 1`, the line-of-sight integrals
                are parallelized using the `multiprocessing` module. Defaults to 1.

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        obstime = validate_user_input(skycoord, obspos)
        earth_xyz = get_earthpos_xyz(obstime, self._ephemeris)
        obs_xyz = get_obspos_xyz(obstime, obspos, earth_xyz, self._ephemeris)

        # Model evaluation is performed in heliocentric ecliptic coordinates. We transform
        # to the barycentric frame, which is compatiable with the Galactic and Celestial,
        # and pretend that that this is the heliocentric frame as we only need the correction
        # rotation.
        skycoord = skycoord.transform_to(coords.BarycentricMeanEcliptic)
        if skycoord.isscalar:
            skycoord_xyz = typing.cast(
                npt.NDArray[np.float64], skycoord.cartesian.xyz.value[:, np.newaxis]
            )
        else:
            skycoord_xyz = typing.cast(npt.NDArray[np.float64], skycoord.cartesian.xyz.value)

        start, stop = get_line_of_sight_range_dicts(
            components=self._ipd_model.comps.keys(),
            unit_vectors=skycoord_xyz,
            obs_pos=obs_xyz,
        )

        instantaneous = obs_xyz.ndim == 1
        if instantaneous:
            obs_xyz = obs_xyz[:, np.newaxis]
            earth_xyz = earth_xyz[:, np.newaxis]

        number_density_partials = self._number_density_partials
        shared_brightness_partial = self._shared_brightness_partial

        dist_coords_to_cores = skycoord.size > nprocesses and nprocesses > 1
        if instantaneous or not dist_coords_to_cores:
            # Populate the instantaneous Earth and observer position in the partial functions.
            number_density_partials = update_partial_earth_pos(
                number_density_partials, earth_pos=earth_xyz
            )
            shared_brightness_partial = functools.partial(shared_brightness_partial, X_obs=obs_xyz)

        emission = np.zeros((self._ipd_model.ncomps, skycoord.size))

        if dist_coords_to_cores:
            skycoord_xyz_splits = np.array_split(skycoord_xyz, nprocesses, axis=-1)
            if not instantaneous:
                # The observer and Earth positions are applied into the partial functions. In the
                # case where we have coordinate-by-coordinate observer positions, we need to ensure
                # that these positions are distributed accordingly over the cores. This means that
                # we need to create partial functions for each core.
                earth_xyz_splits = np.array_split(earth_xyz, nprocesses, axis=-1)
                obs_xyz_splits = np.array_split(obs_xyz, nprocesses, axis=-1)

                number_density_partial_splits = [
                    update_partial_earth_pos(
                        number_density_partials,
                        earth_pos=earth_xyz_split,
                    )
                    for earth_xyz_split in earth_xyz_splits
                ]
                shared_brightness_partial_splits = [
                    functools.partial(shared_brightness_partial, X_obs=obs_xyz_split)
                    for obs_xyz_split in obs_xyz_splits
                ]
            with multiprocessing.get_context(PLATFORM_METHOD).Pool(nprocesses) as pool:
                for idx, comp_label in enumerate(self._ipd_model.comps.keys()):
                    stop_chunks = (
                        [stop[comp_label]] * nprocesses
                        if stop[comp_label].size == 1
                        else np.array_split(stop[comp_label], nprocesses, axis=-1)
                    )
                    start_chunks = (
                        [start[comp_label]] * nprocesses
                        if start[comp_label].size == 1
                        else np.array_split(start[comp_label], nprocesses, axis=-1)
                    )

                    if instantaneous:
                        comp_funcs = [
                            functools.partial(
                                shared_brightness_partial,
                                u_los=skycoord_xyz,
                                start=start,
                                stop=stop,
                                number_density_func=number_density_partials[comp_label],
                                **self._interped_comp_params[comp_label],
                            )
                            for skycoord_xyz, start, stop in zip(
                                skycoord_xyz_splits, start_chunks, stop_chunks
                            )
                        ]
                    else:
                        comp_funcs = [
                            functools.partial(
                                brightness_partial,
                                u_los=skycoord_xyz,
                                start=start,
                                stop=stop,
                                number_density_func=dens_partial[comp_label],
                                **self._interped_comp_params[comp_label],
                            )
                            for skycoord_xyz, start, stop, dens_partial, brightness_partial in zip(
                                skycoord_xyz_splits,
                                start_chunks,
                                stop_chunks,
                                number_density_partial_splits,
                                shared_brightness_partial_splits,
                            )
                        ]
                    proc_chunks = [
                        pool.apply_async(self._integrate_leggauss, args=(func,))
                        for func in comp_funcs
                    ]
                    emission[idx] = np.concatenate([result.get() for result in proc_chunks])

        # Simulate the zodiacal light over the coordinates sequentially.
        else:
            for idx, comp_label in enumerate(self._ipd_model.comps.keys()):
                comp_func = functools.partial(
                    shared_brightness_partial,
                    u_los=skycoord_xyz,
                    start=start[comp_label],
                    stop=stop[comp_label],
                    number_density_func=number_density_partials[comp_label],
                    **self._interped_comp_params[comp_label],
                )
                emission[idx] = self._integrate_leggauss(comp_func)

        emission <<= units.MJy / units.sr
        return emission if return_comps else emission.sum(axis=0)

    def _init_ipd_model_partials(self) -> None:
        """Initialize the partial functions for the interplanetary dust model.

        The spectrally dependant model parameters are interpolated over the provided bandpass or
        delta frequency/wavelength. The partial functions are pre-populated functions that contains
        all non line-of-sight related parameters.
        """
        interp_and_unpack_func = get_model_interp_func(self._ipd_model)
        dicts = interp_and_unpack_func(self._x, self._normalized_weights, self._ipd_model)
        self._interped_comp_params = dicts[0]
        self._interped_shared_params = dicts[1]

        self._shared_brightness_partial = functools.partial(
            self._ipd_model.brightness_at_step_callable,
            bp_interpolation_table=self._b_nu_table,
            **self._interped_shared_params,
        )

        self._number_density_partials = get_partial_number_density_func(comps=self._ipd_model.comps)

    def get_parameters(self) -> dict:
        """Return a dictionary containing the interplanetary dust model parameters.

        This method is mainly meant to be used to fit or sample zodiacal light models.

        Returns:
            parameters: Dictionary of parameters of the interplanetary dust model.
        """
        return self._ipd_model.to_dict()

    def update_parameters(self, parameters: dict) -> None:
        """Update the interplanetary dust model parameters.

        This method is mainly meant to be used to fit or sample zodiacal light models.

        Args:
            parameters: Dictionary of parameters to update. The keys must be the names
                of the parameters as defined in the model. To get the parameters dict
                of an existing model, use the`get_parameters` method of an initialized
                `zodipy.Model`.
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
        self._init_ipd_model_partials()


def validate_user_input(skycoord: coords.SkyCoord, obspos: units.Quantity | str) -> time.Time:
    """Validate the shapes and types of the input coordinate information."""
    try:
        obstime = typing.cast(time.Time, skycoord.obstime)
    except AttributeError as error:
        msg = "The input coordinates must be an astropy SkyCoord object."
        raise TypeError(msg) from error
    if obstime is None:
        msg = "The `obstime` attribute of the `SkyCoord` object must be set."
        raise ValueError(msg)

    try:
        if not isinstance(obspos, str) and obspos.ndim > 1 and obstime.size != obspos.shape[-1]:
            msg = "The number of obstime and obspos must match."
            raise ValueError(msg)
    except AttributeError as error:
        msg = "The observer position must be a string or an astropy Quantity."
        raise TypeError(msg) from error

    if obstime.size > skycoord.size:
        msg = "The number of obstime must be either 1 or the same size as ."
        raise ValueError(msg)

    return obstime
