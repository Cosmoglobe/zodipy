from __future__ import annotations

import functools
import itertools
import multiprocessing
import platform

import numpy as np
import numpy.typing as npt
from astropy import coordinates as coords
from astropy import time, units
from scipy import integrate

from zodipy.blackbody import tabulate_blackbody_emission
from zodipy.bodies import (
    arrange_obstimes,
    get_earthpos_inst,
    get_interp_bodypos,
    get_obspos_from_body,
)
from zodipy.component import ComponentLabel
from zodipy.line_of_sight import (
    get_line_of_sight_range,
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
        x: units.Quantity[units.micron | units.GHz],
        *,
        weights: npt.ArrayLike | None = None,
        name: str = "dirbe",
        gauss_quad_degree: int = 50,
        extrapolate: bool = False,
        ephemeris: str = "builtin",
    ) -> None:
        """Initialize a zodiacal light model.

        Args:
            x: Wavelength or frequency. If `x` is a sequence it is assumed to be a the points
                corresponding to an instrument bandpass and the corresponding `weights` argument
                must be provided.
            weights: Bandpass weights corresponding the the frequencies/wavelengths in `x`. The
                weights are assumed to represent a normalized instrument response in units of
                spectral radiance [Jy/sr].
            name: Zodiacal light model to use. See the
                [docs](https://cosmoglobe.github.io/zodipy/introduction/) for list of available
                models. Defaults to 'dirbe'.
            gauss_quad_degree: Order of the Gaussian-legendre quadrature representing the number of
                discrete points along each line-of-sight. Default is 50 points.
            extrapolate: If `True` all spectral quantities in the selected model are extrapolated to
                the requested frequencies/wavelengths. Else, an exception is raised on values of `x`
                outside of the valid model range. Default is `False`.
            ephemeris: Ephemeris used in Astropy's `solar_system_ephemeris` to compute the positions
                of Earth and optionally the observer. See the
                [Astropy documentation](https://docs.astropy.org/en/stable/coordinates/solarsystem.html)
                for all available ephemerides. Defaults to 'builtin'.

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
        obspos: units.Quantity[units.AU] | str = "earth",
        return_comps: bool = False,
        nprocesses: int = 1,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal light.

        The zodiacal light is simulated for all sky coordinates present in the `skycoord` argument.
        If an obstime and obspos value is not provided for each coordinate value, all coordinates
        are assumed to be observed at an instant from the same position.

        Args:
            skycoord: `astropy.coordinates.SkyCoord` object representing the coordinates for which
                to simulate the zodiacal light. The `obstime` attribute must be specified, and
                correspond to a either a single, or a sequence of observational times, one for each
                coordinate in `skycoord`. The coordinate frame, provided through the `frame` keyword
                in the the `astropy.coordinates.SkyCoord` object (defaults to
                `astropy.coordinates.ICRS`), must be convertible to the
                `astropy.coordinates.BarycentricMeanEcliptic` frame.
            obspos: The heliocentric ecliptic position of the observer, or a string representing an
                observer supported by the `astropy.coordinates.solar_system_ephemeris`. If an
                explicit position is given, it must either be a single, or a sequence of positions,
                one for each coordinate. Defaults to 'earth'.
            return_comps: If `True`, the emission is returned component-wise. Defaults to `False`.
            nprocesses: Number of cores to use. If `nprocesses >= 1`, the line-of-sight integrals
                are distributed and computed in parallel using the `multiprocessing` module.
                Defaults to 1.

        Returns:
            emission: Simulated zodiacal light [MJy/sr].

        """
        try:
            if skycoord.obstime is None:
                msg = "The `obstime` attribute of the `SkyCoord` object is not set."
                raise ValueError(msg)
        except AttributeError as error:
            msg = "The input coordinates must be an astropy SkyCoord object."
            raise TypeError(msg) from error

        try:
            if not (obspos_isstr := isinstance(obspos, str)) and (
                (obspos.ndim > 1 and skycoord.obstime.size != obspos.shape[-1])
                or (obspos.ndim == 1 and skycoord.obstime.size != 1)
            ):
                msg = "The number of obstime (ncoords) and obspos (3, ncoords) does not match."
                raise ValueError(msg)
        except AttributeError as error:
            msg = "The observer position is not a string or an astropy Quantity."
            raise TypeError(msg) from error

        if skycoord.obstime.size > skycoord.size:
            msg = "The size of obstime must be either 1 or ncoords."
            raise ValueError(msg)

        if skycoord.obstime.size == 1:
            interp_obstimes = None
        else:
            interp_obstimes = arrange_obstimes(skycoord.obstime[0].mjd, skycoord.obstime[-1].mjd)

        dist_coords_to_cores = skycoord.size > nprocesses > 1
        if dist_coords_to_cores:
            skycoord_splits = np.array_split(skycoord, nprocesses)
            obspos_splits = (
                itertools.repeat(obspos, nprocesses)
                if obspos_isstr
                else np.array_split(obspos, nprocesses, axis=-1)
            )
            interp_obstime_splits = itertools.repeat(interp_obstimes, nprocesses)
            with multiprocessing.get_context(PLATFORM_METHOD).Pool(nprocesses) as pool:
                emission_splits = [
                    pool.apply_async(self._evaluate, args=(skycoord, obspos, obstime_lims))
                    for skycoord, obspos, obstime_lims in zip(
                        skycoord_splits, obspos_splits, interp_obstime_splits
                    )
                ]
                emission = np.concatenate([split.get() for split in emission_splits], axis=-1)
        else:
            emission = self._evaluate(skycoord, obspos, interp_obstimes)

        emission <<= units.MJy / units.sr
        return emission if return_comps else emission.sum(axis=0)

    def _evaluate(
        self,
        skycoord: coords.SkyCoord,
        obspos: units.Quantity | str,
        interp_obstimes: time.Time | None,
    ) -> npt.NDArray[np.float64]:
        """Evaluate the zodiacal light for a single or a sequence of sky coordinates."""
        if interp_obstimes is None:
            earth_xyz = get_earthpos_inst(skycoord.obstime, self._ephemeris)
        else:
            earth_xyz = get_interp_bodypos(
                body="earth",
                obstimes=skycoord.obstime.mjd,
                interp_obstimes=interp_obstimes,
                ephemeris=self._ephemeris,
            )

        if isinstance(obspos, str):
            obs_xyz = get_obspos_from_body(
                body=obspos,
                obstime=skycoord.obstime,
                interp_obstimes=interp_obstimes,
                earthpos=earth_xyz,
                ephemeris=self._ephemeris,
            )
        else:
            try:
                obs_xyz = obspos.to_value(units.AU)
            except units.UnitConversionError as error:
                msg = "The observer position must be in length units."
                raise units.UnitConversionError(msg) from error

        instantaneous = skycoord.obstime.size == 1
        if instantaneous:
            obs_xyz = obs_xyz[:, np.newaxis]
        if earth_xyz.ndim == 1:
            earth_xyz = earth_xyz[:, np.newaxis]

        # Model evaluation is performed in heliocentric ecliptic coordinates. We transform
        # to the barycentric frame, which is compatiable with the Galactic and Celestial,
        # and pretend that that this is the heliocentric frame as we only need the correction
        # rotation.
        skycoord = skycoord.transform_to(coords.BarycentricMeanEcliptic)

        skycoord_xyz: npt.NDArray[np.float64] = skycoord.cartesian.xyz.value
        if skycoord.isscalar:
            skycoord_xyz = skycoord.cartesian.xyz.value[:, np.newaxis]

        start, stop = get_line_of_sight_range(
            components=self._ipd_model.comps.keys(),
            unit_vectors=skycoord_xyz,
            obs_pos=obs_xyz,
        )

        number_density_partials = self._number_density_partials
        shared_brightness_partial = self._shared_brightness_partial

        number_density_partials = update_partial_earth_pos(
            number_density_partials, earth_pos=earth_xyz
        )
        shared_brightness_partial = functools.partial(shared_brightness_partial, X_obs=obs_xyz)

        emission = np.zeros((self._ipd_model.ncomps, skycoord.size))
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

        return emission

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
        """Return a dictionary containing the zodiacal light model parameters.

        Returns:
            parameters: Zodiacal light model parameter dict.
        """
        return self._ipd_model.to_dict()

    def update_parameters(self, parameters: dict) -> None:
        """Update the zodiacal light model parameters from a parameter dictionary.

        The structure of the input dictionary must match that of the output of the `get_parameters`
        method.

        Args:
            parameters: Zodiacal light model parameter dict.

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
