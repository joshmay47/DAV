"""
This module provides a model for fitting DAV (Deviation Angle Variance) values
to Tropical Cyclone (TC) wind speed with a sigmoid shape.

Originally concieved in: M. F. Pineros, E. A. Ritchie and J. S. Tyo,
"Objective Measures of Tropical Cyclone Structure and Intensity Change From
Remotely Sensed Infrared Image Data" doi: 10.1109/TGRS.2008.2000819.


Example:
-------
        #           a,    b      c    m
        bounds = ( (0,   1000,   0,  1   ),
                   (1e-1,3000,   34, 200 )
                  )
        windows = list(range(6,121,6))
        fill_methods = ["none", "interp", "ffill"] # interp isnt causal
        smooth_methods = ["ewma", "mean", "median", "p20", "min", "last", "weighted"]
        basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

        fits = {}
        for basin in basins:
            print(basin)
            fits[basin] = fit(f'F:/TCs/*/{basin}/*.pkl',
                              bounds,
                              windows,
                              fill_methods,
                              smooth_methods)
Dependencies:
------------
    numpy
    scipy
    tqdm

Author: Joshua May (josh.w.may@gmail.com)
"""

import glob
import pickle
from itertools import product

import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


def fill_dav_values(data: np.array,
                    method: str = "none"):
    """
    Fill in missing (NaN) values in array 'data'.

    data : np.array
        The array that needs values to be filled.
    method : str, optional
        Which method used to fill in NaN values in the TC DAV values. Either
        "none", "interp", or "ffill" for no filling, linear interpolation or
        forward fill, respectively. The default is "none".

    Returns the filled data without NaNs (unless the method is 'none').

    """
    data = np.asarray(data, dtype=float).copy()
    bad = np.isnan(data)

    if bad.all() or method == "none":
        return data

    idx = np.arange(len(data))
    good = ~bad

    if method == "interp":
        # Non-causal: useful as a comparison only
        data[bad] = np.interp(idx[bad], idx[good], data[good])
    elif method == "ffill":
        last = np.nan
        for i, value in enumerate(data):
            if np.isfinite(value):
                last = value
            else:
                data[i] = last
    else:
        raise ValueError(f"Unknown fill method: {method}")

    return data


def causal_smooth(data: np.array,
                  window_size: int = 72,
                  method: str = "weighted",
                  min_periods: str = None,
                  alpha: float = None):
    """
    Smooth DAVs using only values at or before each output time.

    data : np.array
        Array of values to be smoothed.
    window_size : int, optional
        The number of values taken into account when smoothing the DAV values.
        The default is 72.
    method : str, optional
        The method used to smooth. This is either weighted (linearly weighted
        kernel, where the most recent time has the highest weighting,
        window_size periods ago is set to 0) or ewma (exponentially weighted
        kernel, with the same boundary values as weighted, but values
        in-between a scaled according to an exponential shape, rather than
        linear). The default is "weighted". The default is "weighted".
    min_periods : str, optional
        The minimum number of values that must be available before a prediction
        is considered valid. The array returned has the first min_periods set
        to NaN values. The default is None.
    alpha : float, optional
        The alpha value provided to the ewma smoothing method. The default is
        2/(window_size+1).

    Returns the smoothed data.
    """
    data = np.asarray(data, dtype=float)
    out = np.full(len(data), np.nan)

    if min_periods is None:
        min_periods = max(1, window_size // 2)

    if method == "ewma":
        if alpha is None:
            alpha = 2.0 / (window_size + 1.0)
        state = np.nan
        count = 0
        for i, value in enumerate(data):
            if np.isfinite(value):
                count += 1
                state = value if not np.isfinite(
                    state) else alpha * value + (1 - alpha) * state
            if count >= min_periods and np.isfinite(state):
                out[i] = state
        return out

    weights = None
    if method == "weighted":
        weights = np.arange(1, window_size + 1, dtype=float)

    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        values = data[start: i + 1]
        finite = np.isfinite(values)
        finite_values = values[finite]

        if len(finite_values) < min_periods:
            continue

        if method == "mean":
            out[i] = np.mean(finite_values)
        elif method == "median":
            out[i] = np.median(finite_values)
        elif method == "p20":
            out[i] = np.percentile(finite_values, 20)
        elif method == "min":
            out[i] = np.min(finite_values)
        elif method == "last":
            out[i] = finite_values[-1]
        elif method == "weighted":
            local_weights = weights[-len(values):][finite]
            out[i] = np.sum(finite_values * local_weights) / \
                np.sum(local_weights)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    return out


def is_cyclonic(status):
    """Return whether the TC is recorded as cyclonic."""
    return np.isin(status, ("TS", "TY", "ST", "TC", "HU", "HR"))


def load_data(
    pattern: str,
    window: int = 96,
    smoother: str = "weighted",
    fill_method: str = "none",
    min_periods: int = 48,
):
    """
    Load TC data located at glob pattern (pattern).

    pattern : str, optional
        The glob pattern used to retrieve TC files.
    window : int, optional
        The number of values taken into account when smoothing the DAV values.
        The default is 96.
    smoother : str, optional
        The method used to smooth. This is either weighted (linearly weighted
        kernel, where the most recent time has the highest weighting,
        window_size periods ago is set to 0) or ewma (exponentially weighted
        kernel, with the same boundary values as weighted, but values
        in-between a scaled according to an exponential shape, rather than
        linear). The default is "weighted".
    fill_method : str, optional
        Which methods tested to fill in NaN values in the TC DAV values. Either
        "none", "interp", or "ffill" for no filling, linear interpolation or
        forward fill, respectively. The default is "none".
    min_periods : int, optional
        The minimum number of values that must be available before a prediction
        is considered valid. The array returned has the first min_periods set
        to NaN values. The default is 48.

    Returns
    -------
    xs: np.array
        Filtered values for TC DAV value.
    ys: np.array
        Filtered values for TC wind.
    groups: np.array
        Array where indices correspond to xs and ys indices, and values
        are which idx from storm is used for the value.
    storms : list
        List of individual TC files used.

    """
    xs, ys, groups, storms = [], [], [], []

    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            tc = pickle.load(f)

        n = min(
            len(tc["usa_wind"]),
            len(tc["davs"]),
            len(tc["usa_status"]),
            len(tc["dist2land"]),
        )

        if n < 48:
            continue

        dav = fill_dav_values(tc["davs"][:n], method=fill_method)
        if np.isnan(dav).all():
            continue

        dav = causal_smooth(dav, window_size=window,
                            method=smoother, min_periods=min_periods)
        wind = np.asarray(tc["usa_wind"], dtype=float)[:n]
        status = np.asarray(tc["usa_status"])[:n]
        dist2land = np.asarray(tc["dist2land"], dtype=float)[:n]

        valid = (
            np.isfinite(dav)
            & np.isfinite(wind)
            & is_cyclonic(status)
            & (dist2land >= 300)
        )

        if valid.any():
            storm_id = len(storms)
            xs.append(dav[valid])
            ys.append(wind[valid])
            groups.extend([storm_id] * int(valid.sum()))
            storms.append(path)

    return np.concatenate(xs), np.concatenate(ys), np.asarray(groups), storms


def sigmoid(x, a, b, c, m):
    """Sigmoid function.

    Clipping exists so enormous or infinitesimal values don't get generated.
    """
    return c + m / (1.0 + np.exp(np.clip(a * (x - b), -60, 60)))


def rmse(y_true, y_pred):
    """Root mean square error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_sigmoid(x: np.array,
                  y: np.array,
                  groups: list,
                  bounds: tuple,
                  p0: tuple):
    """
    Fits a sigmoid given filtered x and y values using Cross validation.

    This does not return parameters, as it is primarily used to find the best
    hyperparameters. Use scipy.optimize.curve_fit for the parameters.

    x : array
        Filtered x values, DAV values in this case. (units of deg^2, typically
        between 500-3000)
    y : array
        Filtered y values, wind speed values in this case. (units of knots,
        typically between 20 and 180).
    groups : list(int)
        Which indices of x and y correspond to which TC, must be the same
        size as x and y.
    bounds : tuple(tuple)
        A tuple of shape (2,4) that contains the upper (first index) and lower
        (second index) bounds for the sigmoid parameters (a,b,c,m)
    p0 : tuple
        A tuple of shape (4) that contains the initial guesses for sigmoid
        parameters a,b,c,m.


    """
    cv_pred = np.empty_like(y)
    unique_groups = np.unique(groups)
    folds = [unique_groups[i::10] for i in range(10)]

    for fold_groups in folds:
        test = np.isin(groups, fold_groups)
        train = ~test

        fold_params, * \
            _ = curve_fit(sigmoid, x[train], y[train], p0=p0, bounds=bounds)
        cv_pred[test] = sigmoid(x[test], *fold_params)

    cv_rmse = rmse(y, cv_pred)
    abs_error = np.abs(y - cv_pred)
    cv_mae = np.mean(abs_error)
    p_good = float(np.mean(abs_error <= 15) * 100)
    return {"RMSE": cv_rmse,
            "MAE": cv_mae,
            "Proportion_good": p_good}


def fit(tc_file_pattern: str,
        bounds: tuple,
        windows: list,
        fill_methods: str,
        smooth_methods: str) -> dict:
    """
    Fits TCs (at tc_file_pattern) to the optimal sigmoid shape given arguments.

    tc_file_pattern : str
        glob pattern to retrieve the relevant TC files.
    bounds : tuple(tuple)
        A tuple of shape (2,4) that contains the upper (first index) and lower
        (second index) bounds for the sigmoid parameters (a,b,c,m)
    windows : list(int)
        A list of smoothing windows that are to be tested.
    fill_methods : str
        Which methods tested to fill in NaN values in the TC DAV values. Either
        "none", "interp", or "ffill" for no filling, linear interpolation or
        forward fill, respectively.
    smooth_methods : str
        The methods tested to smooth. This is either weighted (linearly
        weighted kernel, where the most recent time has the highest weighting,
        window_size periods ago is set to 0) or ewma (exponentially weighted
        kernel, with the same boundary values as weighted, but values
        in-between a scaled according to an exponential shape, rather than
        linear).

    Returns the optimal sigmoid parameters and filling and smoothing methods
    for the TCs located in the tc_file_pattern as a dict.

    """
    # Initial guess is just the middle value of bounds
    p0 = tuple(0.5*(lower_bound+upper_bound)
               for lower_bound, upper_bound in zip(*bounds))

    n_tests = len(windows)*len(fill_methods)*len(smooth_methods)

    best_rmse = np.inf
    best_hyperparams = None
    for window, fill_method, smooth_method in tqdm(product(windows,
                                                           fill_methods,
                                                           smooth_methods),
                                                   total=n_tests):
        x, y, groups, storms = load_data(pattern=tc_file_pattern,
                                         window=window,
                                         smoother=smooth_method,
                                         fill_method=fill_method,
                                         min_periods=min(window, 48))  # Going over 48 (24 hr) is excessive
        result = train_sigmoid(x, y, groups, bounds, p0)
        if result['RMSE'] < best_rmse:
            best_rmse = result['RMSE']
            best_hyperparams = (window, fill_method, smooth_method)

    # Fit with all data given the best found hyperparameters.
    x, y, groups, storms = load_data(pattern=tc_file_pattern,
                                     window=best_hyperparams[0],
                                     smoother=best_hyperparams[2],
                                     fill_method=best_hyperparams[1],
                                     min_periods=min(best_hyperparams[0], 48))
    params, *_ = curve_fit(sigmoid, x, y, p0=p0, bounds=bounds)
    return {"parameters": {param: val for param, val in zip("abcm",
                                                            params)},
            "window": best_hyperparams[0],
            "smooth_method": best_hyperparams[2]}


if __name__ == "__main__":
    #           a,    b      c    m
    bounds = ((0,   1000,   0,  1),
              (1e-1, 3000,   34, 200)
              )
    windows = list(range(6, 121, 6))
    fill_methods = ["none", "interp", "ffill"]  # interp isnt causal
    smooth_methods = ["ewma", "mean", "median",
                      "p20", "min", "last", "weighted"]
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]

    fits = {}
    for basin in basins:
        print(basin)
        fits[basin] = fit(fr"F:\TCs\*\{basin}\*.pkl",
                          bounds,
                          windows,
                          fill_methods,
                          smooth_methods)
