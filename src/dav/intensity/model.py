"""
This module provides a model for predicting Tropical Cyclone (TC) wind speed
from DAV (Deviation Angle Variance) values.

Originally concieved in: M. F. Pineros, E. A. Ritchie and J. S. Tyo,
"Objective Measures of Tropical Cyclone Structure and Intensity Change From
Remotely Sensed Infrared Image Data" doi: 10.1109/TGRS.2008.2000819.

Workflow:
    raw DAVs → smoothing → sigmoid model → wind prediction

Supported basins:
    EP, NA, NI, SI, SP, WP

Dependencies:
    numpy

Example:
    import numpy as np
    from wind_sigmoid_model import predict

    example_davs = 1000*np.random.random(size=400)+2000
    example_prediction = predict(example_davs, "EP")

Notes:
    - NaN values in input are ignored during smoothing.
    - Output values are NaN until min_periods (argument of the smooth function)
        valid observations are available.

Author: Joshua May (josh.w.may@gmail.com)
"""

import numpy as np

MODEL_PARAMS = {'EP': {'parameters': {'a': 0.00433131558382244,
                                      'b': 1710.198025712262,
                                      'c': 34.0,
                                      'm': 83.609866065916},
                       'window': 42,
                       'smooth_method': 'ewma'},

                'NA': {'parameters': {'a': 0.0030938597674090573,
                                      'b': 1573.17568622581,
                                      'c': 34.0,
                                      'm': 119.19126322586484},
                       'window': 96,
                       'smooth_method': 'weighted'},

                'NI': {'parameters': {'a': 0.003173214957622463,
                                      'b': 1475.3089347608204,
                                      'c': 34.0,
                                      'm': 150.12901578182283},
                       'window': 84,
                       'smooth_method': 'weighted'},

                'SI': {'parameters': {'a': 0.0042467627405537185,
                                      'b': 1667.2522138005518,
                                      'c': 34.0,
                                      'm': 98.12715860124835},
                       'window': 54,
                       'smooth_method': 'ewma'},

                'SP': {'parameters': {'a': 0.0036663525428644907,
                                      'b': 1619.260661303375,
                                      'c': 34.0,
                                      'm': 98.88908060526117},
                       'window': 96,
                       'smooth_method': 'weighted'},

                'WP': {'parameters': {'a': 0.0035016934408498524,
                                      'b': 1556.1076509000422,
                                      'c': 34.0,
                                      'm': 122.45230275464765},
                       'window': 72,
                       'smooth_method': 'weighted'}
                }

__all__ = ["predict", "smooth", "sigmoid"]

DEFAULT_WINDOW = 90
DEFAULT_MIN_PERIODS = 45
DEFAULT_METHOD = "weighted"


def smooth(davs: np.ndarray,
           window_size: int = DEFAULT_WINDOW,
           min_periods: int = DEFAULT_MIN_PERIODS,
           method: str = DEFAULT_METHOD) -> np.ndarray:
    """Smooth DAVs using only values at or before each output time.

    There are various smoothing parameters that can be tuned. The optimal ones
    for each basin are stored as hyperparameters in MODEL_PARAMS.

    davs: array of floats
        DAV values of a single TC.
    window_size: integer
        The number of values taken into account when smoothing the DAV values.
    min_periods: integer
        The minimum number of values that must be available before a prediction
        is considered valid. The array returned has the first min_periods set
        to NaN values.
    method: str
        The method used to smooth. This is either weighted (linearly weighted
        kernel, where the most recent time has the highest weighting,
        window_size periods ago is set to 0) or ewma (exponentially weighted
        kernel, with the same boundary values as weighted, but values
        in-between a scaled according to an exponential shape, rather than
        linear).
        """
    data = np.asarray(davs, dtype=float)
    out = np.full(len(davs), np.nan)

    if method == "ewma":
        alpha = 2.0 / (window_size + 1.0)
        state = np.nan
        count = 0
        for i, value in enumerate(data):
            if np.isfinite(value):
                count += 1
                state = value if not np.isfinite(state) else alpha * value + (1 - alpha) * state # Initialize EWMA with first finite value
            if count >= min_periods and np.isfinite(state):
                out[i] = state

    elif method == "weighted":
        weights = np.arange(1, window_size + 1, dtype=float)
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            values = data[start:i + 1]
            finite = np.isfinite(values)
            finite_values = values[finite]

            if len(finite_values) < min_periods:
                continue
            local_weights = weights[-len(values):][finite]
            out[i] = np.sum(finite_values * local_weights) / np.sum(local_weights)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

    return out


def sigmoid(smoothed_dav: np.ndarray,
            params: dict[str, float]) -> np.ndarray:
    """Predict wind from already-smoothed DAV values using a sigmoid function.

    smoothed_dav: array of floats
        Array of smoothed TC DAV values.
    params: dict of (str, float), (key, value) pairs.
        Required keys: 'a', 'b', 'c', 'm'
        Contains the variables for the sigmoid function.
        a = sigmoid sharpness
        b = horizontal shift
        c = vertical shift
        m = sigmoid magnitude.
    """
    x = np.asarray(smoothed_dav, dtype=float)
    a = params["a"]
    b = params["b"]
    c = params["c"]
    m = params["m"]

    return c + m / (1.0 + np.exp(np.clip(a * (x - b), -60, 60)))


def predict(davs: np.ndarray,
            basin: str,
            min_periods: int = DEFAULT_MIN_PERIODS) -> np.ndarray:
    """Predict wind from raw DAV values, given a basin.

    Ensure the function is called separately for each TC as the signal will be
    smoothed and boundary values are considered.

    davs : array of floats
        Array of a single TCs DAV values in deg^2. Typically expecting values
        in the range of 500-3000.
    basin : str
        Basin of the TCs location must be: 'EP', 'NA', 'NI', 'SI', 'SP', or 'WP'.
    min_periods : int
        The number of periods before a confident prediction is reached. This is
        done to allow the smoothing method to retrieve enough earlier values.
        Unit is half-hours.

    Returns: np.ndarray
        Array of predicted wind speeds (same length as input) in knots.
        Values before min_periods are NaN.
    """
    try:
        davs = np.asarray(davs, dtype=float)
    except ValueError:
        raise ValueError("DAV values must be numerical in an array.")
    if basin not in MODEL_PARAMS:
        raise ValueError(f"Basin must be in {MODEL_PARAMS.keys()}, got {basin!r}.")
    if len(davs.shape) != 1:
        raise ValueError(f"DAV values must be 1D array, got shape: {davs.shape}.")

    model = MODEL_PARAMS[basin]
    smoothed_davs = smooth(davs,
                           window_size=model['window'],
                           method=model['smooth_method'],
                           min_periods=min_periods)
    prediction = sigmoid(smoothed_davs, model['parameters'])
    return prediction
