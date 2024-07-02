from __future__ import annotations

import functools
import multiprocessing
import multiprocessing.pool
import platform
import typing

import numpy as np
import numpy.typing as npt
from astropy import coordinates as coords
from astropy import time, units
from scipy import integrate

from zodipy.blackbody import tabulate_blackbody_emission
from zodipy.bodies import get_earthpos_xyz, get_obspos_xyz
from zodipy.line_of_sight import (
    get_line_of_sight_range,
    integrate_leggauss,
)
from zodipy.model_registry import model_registry
from zodipy.number_density import populate_number_density_with_model
from zodipy.unpack_model import get_model_to_dicts_callable
from zodipy.zodiacal_component import ComponentLabel

_PLATFORM_METHOD = "fork" if "windows" not in platform.system().lower() else None


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
                msg = "Several wavelengths are provided by no weights."
                raise ValueError(msg)
        except AttributeError as error:
            msg = "The input 'x' must be an astropy Quantity."
            raise TypeError(msg) from error
        if x.isscalar and weights is not None:
            msg = "A single wavelength is provided with weights."
            raise ValueError(msg)

        self._interplanetary_dust_model = model_registry.get_model(name)

        if not extrapolate and not self._interplanetary_dust_model.is_valid_at(x):
            msg = (
                "The requested frequencies are outside the valid range of the model. "
                "If this was intended, set the extrapolate argument to True."
            )
            raise ValueError(msg)

        bandpass_is_provided = weights is not None
        if bandpass_is_provided:
            weights = np.asarray(weights)
            if x.size != weights.size:
                msg = "Number of wavelengths and weights must be the same in the bandpass."
                raise ValueError(msg)
            normalized_weights = weights / integrate.trapezoid(weights, x)
        else:
            normalized_weights = None
        self._b_nu_table = tabulate_blackbody_emission(x, normalized_weights)

        # Interpolate and convert the model parameters to dictionaries which can be used to evaluate
        # the zodiacal light model.
        brightness_callable_dicts = get_model_to_dicts_callable(self._interplanetary_dust_model)(
            x, normalized_weights, self._interplanetary_dust_model
        )
        self._comp_parameters, self._common_parameters = brightness_callable_dicts

        self._ephemeris = ephemeris
        self._leggauss_points_and_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)

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
        be observed from that position at that time. Otherwise, when `obspos` and `obstime` contains
        multiple values, corresponding to coordinates in `skycoord`, the zodiacal light is simulated
        in a time-ordered manner.

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

        skycoord = skycoord.transform_to(coords.BarycentricMeanEcliptic)
        if skycoord.isscalar:
            skycoord_xyz = typing.cast(
                npt.NDArray[np.float64], skycoord.cartesian.xyz.value[:, np.newaxis]
            )
        else:
            skycoord_xyz = typing.cast(npt.NDArray[np.float64], skycoord.cartesian.xyz.value)

        start, stop = get_line_of_sight_range(
            components=self._interplanetary_dust_model.comps.keys(),
            unit_vectors=skycoord_xyz,
            obs_pos=obs_xyz,
        )

        # If a time and obspos is provided per coordinate, we need to make arrays broadcastable.
        if obs_xyz.ndim == 1:
            obs_xyz = obs_xyz[:, np.newaxis]
            earth_xyz = earth_xyz[:, np.newaxis]

        # Return a dict of partial functions corresponding to the number density of each zodiacal
        # component in the interplanetary dust model.
        density_callables = populate_number_density_with_model(
            comps=self._interplanetary_dust_model.comps,
            dynamic_params={"X_earth": earth_xyz},
        )

        # Create partial function of the brightness integral at a step along the line-of-sight with
        # shared arguments between zodiacal components.
        common_integrand = functools.partial(
            self._interplanetary_dust_model.brightness_at_step_callable,
            X_obs=obs_xyz,
            bp_interpolation_table=self._b_nu_table,
            **self._common_parameters,
        )

        emission = np.zeros((self._interplanetary_dust_model.ncomps, skycoord.size))
        dist_to_cores = skycoord.size > nprocesses and nprocesses > 1
        if dist_to_cores:
            skycoord_xyz_splits = np.array_split(skycoord_xyz, nprocesses, axis=-1)
            with multiprocessing.get_context(_PLATFORM_METHOD).Pool(nprocesses) as pool:
                for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                    stop_chunks = np.array_split(stop[comp_label], nprocesses, axis=-1)
                    if start[comp_label].size == 1:
                        start_chunks = [start[comp_label]] * nprocesses
                    else:
                        start_chunks = np.array_split(start[comp_label], nprocesses, axis=-1)
                    comp_integrands = [
                        functools.partial(
                            common_integrand,
                            u_los=vec,
                            start=start,
                            stop=stop,
                            get_density_function=density_callables[comp_label],
                            **self._comp_parameters[comp_label],
                        )
                        for vec, start, stop in zip(skycoord_xyz_splits, start_chunks, stop_chunks)
                    ]

                    proc_chunks = [
                        pool.apply_async(
                            integrate_leggauss,
                            args=(comp_integrand, *self._leggauss_points_and_weights),
                        )
                        for comp_integrand in comp_integrands
                    ]
                    emission[idx] = np.concatenate([result.get() for result in proc_chunks])

        else:
            for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                comp_integrand = functools.partial(
                    common_integrand,
                    u_los=skycoord_xyz,
                    start=start[comp_label],
                    stop=stop[comp_label],
                    get_density_function=density_callables[comp_label],
                    **self._comp_parameters[comp_label],
                )
                emission[idx] = integrate_leggauss(
                    comp_integrand, *self._leggauss_points_and_weights
                )

        emission <<= units.MJy / units.sr
        return emission if return_comps else emission.sum(axis=0)

    def get_parameters(self) -> dict:
        """Return a dictionary containing the interplanetary dust model parameters.

        This method is mainly meant to be used to fit or sample zodiacal light models.

        Returns:
            parameters: Dictionary of parameters of the interplanetary dust model.
        """
        return self._interplanetary_dust_model.to_dict()

    def update_parameters(self, parameters: dict) -> None:
        """Update the interplanetary dust model parameters.

        This method is mainly meant to be used to fit or sample zodiacal light models.

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
                        self._interplanetary_dust_model.comps[ComponentLabel(comp_key)]
                    )(**comp_value)
            elif isinstance(value, dict):
                _dict[key] = {ComponentLabel(k): v for k, v in value.items()}

        self._interplanetary_dust_model = self._interplanetary_dust_model.__class__(**_dict)


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
