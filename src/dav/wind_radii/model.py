"""
Estimate tropical cyclone (TC) wind radii using polynomial models trained on
derived atmospheric and environmental features.

The models estimate a specified TC wind radius (e.g. 34-, 50-, or 64-knot
winds) from a combination of:

* Distance from the TC centre at which the DAV profile falls below a
  specified threshold.
* Maximum sustained wind speed (VMax).
* Sea surface temperature (SST).
* Time since the TC last reached 34 kt intensity.

Pre-trained models are provided with the package. Users can select a model
using its basin, quadrant, and wind-radius parameters without needing to
interact directly with the underlying model files.

The DAV profile is expected to contain four radial profiles corresponding to
the northeast, southeast, southwest, and northwest quadrants. A symmetric
profile can also be derived by combining the four quadrants.

The methodology for deriving tropical cyclone structural information from
remotely sensed infrared imagery was originally described in:

    Dolling, K., Ritchie, E. and Tyo, J. (2016). The Use of the Deviation Angle
    Variance Technique on Geostationary Satellite Imagery to Estimate Tropical
    Cyclone Size Parameters. Weather and Forecasting 31(5) pp. 1625-1642.
    https://journals.ametsoc.org/view/journals/wefo/31/5/waf-d-16-0056_1.xml

Example
-------
A pre-trained model can be used to predict wind radii from a collection of
observations::

    data = {"profile": profile,
            "age": age,
            "sst": sst,
            "wind": wind}

    radii = predict(data,
                    basin="NA",
                    quadrant="symmetric",
                    radius="r34")

Here, each input array contains observations for one or more time steps and
``profile`` has shape ``(N, 4, M)``, where ``N`` is the number of observations
and ``M`` is the number of radial pixels.

The ``basin``, ``quadrant``, and ``radius`` arguments select the corresponding
pre-trained model supplied with the package. The model files are loaded
automatically and do not need to be accessed directly.

Dependencies
------------
numpy

Author: Joshua May (josh.w.may@gmail.com)
"""

import numpy as np
import warnings
from dataclasses import dataclass
from importlib.resources import files
import json

from .preprocessing import quadrants_to_symmetrical
from ..utils.polar_feature_extractor import DAVProfile


BASINS = {"EP", "NA", "SI", "NI", "SP", "WP"}
QUADRANTS = ("symmetric", "ne", "se", "sw", "nw")
QUADRANT_INDEX = {"ne": 0, "se": 1, "sw": 2, "nw": 3}
RADII = {"r34": 34, "r50": 50, "r64": 64}


@dataclass(frozen=True)
class DataRange:
    low: float
    high: float
    unit: str

    def in_range(self, values):
        values = np.asarray(values)
        return np.isnan(values) | ((self.low <= values) & (values <= self.high))

    def warn_if_out_of_range(self, values, name="value"):
        valid = self.in_range(values)
        if not np.all(valid):
            bad = values[~valid]
            warnings.warn(f"{name} contains {len(bad)} values outside the "
                          f"expected range {self.low}-{self.high} "
                          f"{self.unit}. Examples: {bad[:5]}",
                          UserWarning,
                          stacklevel=2)


# Given by lowest and highest recorded values in history
EXPECTED_DATA_RANGES = {"profile": DataRange(low=0, high=8100, unit="deg^2"),
                        "sst":     DataRange(low=0, high=50,   unit="degC"),
                        "wind":    DataRange(low=0, high=220,  unit="kt"),
                        "age":     DataRange(low=0, high=900,  unit="hours")}


def _furthest_contiguous_under_threshold(array, threshold):
    """Retrieve the number of pixels that stay under threshold along axis 1 contiguously from the start."""
    if array.ndim != 2:
        raise ValueError(f"Invalid shape for profile, should be 2D, got {array.ndim}D.")
    below = array < threshold
    connected = below.cumprod(axis=1)
    counts = connected.sum(axis=1)
    return counts


def get_dav_radii_from_profile(profile: DAVProfile,
                               quadrant: str,
                               threshold: float) -> np.ndarray:
    """
    Convert the DAV profile to dav_radii in a quadrant using a threshold.

    Parameters
    ----------
    profile : np.ndarray
        array of shape (N, 4, M) with DAV values across N time-steps M pixels
        away from the centre in four directions (North East, South East,
                                                 South West, North West)
    quadrant : str
        the quadrant the profile is to be extracted from. Should be one of
        ("symmetric", "ne", "se", "sw", "nw").
    threshold : float
        the value that the profile should stay under to be considered part of
        dav_radii.

    Raises
    ------
    ValueError
        When the profile is not of shape (N, 4, M).

    Returns
    -------
    np.ndarray
        dav_radii, the distance from the center of the TC the DAV value stayed
        below the threshold in the given quadrant in KM. Ensure the
        profile.resolution matches the profile resolution in km/pixel

    """
    if profile.values.ndim != 3:
        raise ValueError("Invalid shape for profile, should be 3D, "
                         f"got {profile.values.ndim}D.")
    if profile.values.shape[1] != 4:
        raise ValueError("Invalid number of quadrants in profile, should be "
                         f"4, got {profile.values.shape[1]}.")
    if quadrant not in QUADRANTS:
        raise ValueError(f"Invalid quadrant {quadrant!r}. Expected one of "
                         f"{QUADRANTS}.")

    if quadrant == "symmetric":
        quadrant_profile = quadrants_to_symmetrical(profile.values)
    else:
        quadrant_profile = profile.values[:, QUADRANT_INDEX[quadrant]]

    dist_pixels = _furthest_contiguous_under_threshold(quadrant_profile,
                                                       threshold)
    dist_km = profile.resolution*dist_pixels
    return dist_km


def _check_data_inputs(model_json: dict,
                       data: dict,
                       variables: list):
    """Reorders the data and checks validity along the way."""
    missing = [v for v in variables if v not in data and v != "dav_radius"]

    if missing:
        raise ValueError(f"Missing required input variables: {missing}")

    dav_radius = get_dav_radii_from_profile(data["profile"],
                                            model_json["quadrant"],
                                            model_json["dav_radius_threshold"])

    feature_data = {**data,
                    "dav_radius": dav_radius}

    n = data["profile"].values.shape[0]
    for name in variables:
        name_length = np.asarray(feature_data[name]).shape[0]
        if name_length != n:
            raise ValueError("Time dimensions aren't equal: Profile length ("
                             f"{n}). {name} length ({name_length}).")

    for name, expected_range in EXPECTED_DATA_RANGES.items():
        if name == "profile":
            feature = data["profile"].values
        else:
            feature = data[name]
        expected_range.warn_if_out_of_range(feature, name)

    # Build X, this is an array of shape (N, 4), columns are variables.
    X = [np.asarray(feature_data[var]) for var in variables]
    X = np.column_stack(X)
    return X


def predict_from_json(model_json: dict, data: dict) -> np.array:
    """
    Predict TC wind radii based on supplied model and data.

    Parameters
    ----------
    model_json : dict
        JSON that contains model info, should have intercept, terms, variables
        at a minimum.
        Is generated by pipeline = fit(); json = pipeline_to_json(pipeline)
    data : dict
        dictionary with keys (profile, age, sst, wind)
        profile : DAVProfile object
            values : np.ndarray of shape (N, 4, M).
                View of surrounding DAV pixels.
            resolution : float
                Resolution of profile in km/pixel.
            [Profile is generated with dav.utils.get_dav_profile()]
        age : np.array of shape (N)
            Time since TC last reached 34 kt intensity in hours. Generated by
            dav.utils.get_tc_age()
        sst : np.array of shape (N)
            Sea surface temperature at location of TC, in degrees Celcius
        wind : np.array of shape (N)
            Also known as VMax, maximum wind speed of TC at given times, in kt.

    Raises
    ------
    ValueError
        When the json or data does not contain expected keywords.

    Returns
    -------
    y : np.array
        Wind radius given with the basin, quadrant, and radius specified by the
        model_json.

    """
    intercept = model_json["intercept"]
    terms = model_json["terms"]
    variables = model_json["variables"]

    X = _check_data_inputs(model_json, data, variables)
    N = X.shape[0]

    # y = intercept + sum_across_n(coeff_n * input1^power2 * input2^power2...)
    y = np.full(N, fill_value=intercept)
    for term in terms:
        coef = term["coef"]
        powers = term["powers"]

        term_val = np.ones(N)
        for j in np.nonzero(powers)[0]:
            term_val *= X[:, j] ** powers[j]

        y += coef * term_val

    return y


def predict(data: dict,
            basin: str,
            quadrant: str,
            radius: str,
            set_low_zero: bool = True) -> np.ndarray:
    """
    Predict TC wind radii based on default model and data.

    Parameters
    ----------
    data : dict
        dictionary with keys (profile, age, sst, wind)
        profile : DAVProfile object
            values : np.ndarray of shape (N, 4, M).
                View of surrounding DAV pixels.
            resolution : float
                Resolution of profile in km/pixel.
            [Profile is generated with dav.utils.get_dav_profile()]
        age : np.array of shape (N)
            Time since TC last reached 34 kt intensity in hours, generated by
            utils.get_tc_age()
        sst : np.array of shape (N)
            Sea surface temperature at location of TC, in degrees Celcius
        wind : np.array of shape (N)
            Also known as VMax, maximum wind speed of TC at given times, in kt.
    basin : str
        The basin the storm is located in. If there are multiple basins, it
        should be in the basin of genesis. Should be one of
        ("EP", "NA", "SI", "NI", "SP", "WP").
    quadrant : str
        The wind quadrant to be modelled. Should be one of
        ("symmetric", "ne", "se", "sw", "nw").
    radius : str
        The wind value to predict the radius for. Should be one of
        ("r34", "r50", "r64").
    set_low_zero : bool
        If true, sets values of prediction to zero when the wind value is less
        than the desired intensity. Default is True.

    Raises
    ------
    ValueError
        When the json or data does not contain expected keywords.
        When the basin, quadrant or radius are invalid

    Returns
    -------
    y : np.array
        Wind radius given with the basin, quadrant, and radius.

    """
    errors = ""
    if basin not in BASINS:
        errors += f"Basin should be one of: {BASINS}, got {basin}.\n"
    if quadrant not in QUADRANTS:
        errors += f"Quadrant should be one of: {QUADRANTS}, got {quadrant}.\n"
    if radius not in RADII:
        errors += f"Radius should be one of: {RADII}, got {radius}.\n"
    if errors:
        raise ValueError(errors[:-1])

    filename = f"MODEL_{basin}_{quadrant}_{radius}.json"
    model_file = (files("dav.wind_radii")
                  / "models"
                  / filename)

    with model_file.open("r", encoding="utf-8") as f:
        model_json = json.load(f)

    prediction = predict_from_json(model_json, data)

    if set_low_zero:
        return np.where(data['wind'] < RADII[radius], 0., prediction)

    return prediction
