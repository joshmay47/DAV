"""
Train and serialize tropical cyclone wind-radius estimation models.

This module provides the data-processing, feature-engineering, model-training,
evaluation, and serialization pipeline used to generate the JSON model
definitions consumed by the prediction functions in the package.

Models are trained separately for each combination of:

* Tropical cyclone basin.
* Radial quadrant (symmetric, northeast, southeast, southwest, or northwest).
* Wind-radius threshold (34, 50, or 64 kt).

The default model uses the following predictors:

* ``dav_radii``: the radial extent of the DAV profile below a fitted
  threshold, in km.
* ``age``: storm age since the tropical cyclone last reached 34 kt, in hours.
* ``sst``: sea surface temperature at the tropical cyclone centre, in °C.
* ``wind``: maximum sustained wind speed, in knots.

Input tropical cyclone data is loaded from pickle files, validated, filtered
for quality and relevance, and converted into training data. DAV profiles are
smoothed before the radial DAV feature is calculated. The training pipeline
uses polynomial feature expansion followed by standardisation and either
Ridge or Lasso regression.

Hyperparameters are selected using grouped cross-validation, with tropical
cyclone storm IDs used as groups to prevent observations from the same storm
appearing in both training and validation folds. A separate grouped
train/test split is used for final model evaluation.

The fitted scikit-learn pipeline is converted into a compact JSON
representation containing the polynomial model coefficients, predictor
metadata, fitted DAV threshold, model configuration, creation timestamp, and
evaluation metrics. This representation is designed to be independent of
scikit-learn and can be evaluated by the prediction module.

Data structure
--------------
The input dataset consists of one pickle file per tropical cyclone. Each file
contains a dictionary with the following fields::

    basin       : str
    sid         : str
    name        : str
    season      : int
    davs        : (N,) array
    dav_radii   : (N, 4, M) array
    age         : (N,) array
    usa_wind    : (N,) array
    usa_r34     : (N, 4) array
    usa_r50     : (N, 4) array
    usa_r64     : (N, 4) array
    sst         : (N,) array
    usa_status  : list of length N
    dist2land   : (N,) array

Here ``N`` is the number of observations for the storm and ``M`` is the
number of radial DAV pixels. The four columns of the radial fields correspond
to the northeast, southeast, southwest, and northwest quadrants.

Model output
------------
Each fitted model is serialized as a JSON file containing, among other
metadata:

* The basin, quadrant, and wind-radius target.
* The input variables and their units.
* The selected DAV-radius threshold.
* The polynomial model intercept and coefficients.
* Training and test evaluation metrics.
* The time at which the model was generated.

The resulting JSON files are intended to be consumed by the model prediction
module rather than used directly as scikit-learn model objects.

Notes
-----
The default profile resolution is 8 km per pixel. DAV-radius thresholds are
specified in the same units as the DAV profile.

Wind-radius observations are converted from nautical miles (IBTrACS default) to
kilometres before model fitting.

The model currently supports Ridge and Lasso regression. Hyperparameters
including polynomial degree, regression strength, and DAV threshold can be
searched using :func:`fit_with_search`.

Dependencies
------------
numpy
pandas
scipy
tqdm
psutil
scikit-learn


Author: Joshua May (josh.w.may@gmail.com)
"""

import pickle
import pandas as pd
from pathlib import Path
from tqdm.contrib.itertools import product as tqdm_product
from itertools import product
import numpy as np
from scipy import ndimage
import psutil
import os
import json
from datetime import datetime, UTC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.dummy import DummyRegressor

from .preprocessing import smooth_dav, forward_fill_sst, quadrants_to_symmetrical

BASINS = ("EP", "NA", "NI", "SI", "SP", "WP")
QUADRANTS = ("symmetric", "ne", "se", "sw", "nw")
QUADRANT_INDEX = {"ne": 0, "se": 1, "sw": 2, "nw": 3}
RADII = {"r34": "usa_r34", "r50": "usa_r50", "r64": "usa_r64"}

DEFAULT_DAV = 2700  # Statistical default for a uniform distribution (bad image)
PROFILE_RESOLUTION = 8  # In km per pixel
NM_KM = 1.852  # Conversion ratio between nautical miles and kilometers.
MEMORY_SAFETY_FACTOR = 2.0


VARIABLE_METADATA = {
    "dav_radius": {
      "description": "Furthest distance from TC center that stays below the dav_radius_threshold",
      "units": "km"
    },
    "age": {
      "description": "Storm age since last reaching 34 knots wind speed",
      "units": "hours"
    },
    "sst": {
      "description": "Sea surface temperature at TC center",
      "units": "degC"
    },
    "wind": {
      "description": "Maximum sustained wind speed at current TC time",
      "units": "knots"
    }
  }

TC_STRUCTURE = """
TC data structure
-----------------
basin       : str
sid         : str
davs        : (n,) array
dav_radii   : (n, 4, k) array
age         : (n,) array
usa_wind    : (n,) array
usa_r34     : (n, 4) array
usa_r50     : (n, 4) array
usa_r64     : (n, 4) array
sst         : (n,) array
usa_status  : list, length n
dist2land   : (n,) array

All time-dependent fields must have the same length n.
The four columns of the radial fields correspond to the
four quadrants. k corresponds to the number of pixels
the dav_radii reaches out to.
"""


def validate_tc_structure(tc, raise_on_error=True):
    """Validate that a TC dictionary has the structure required by this package."""

    errors = []

    required = {"basin": str,
                "davs": np.ndarray,
                "dav_radii": np.ndarray,
                "age": np.ndarray,
                "usa_wind": np.ndarray,
                "usa_r34": np.ndarray,
                "usa_r50": np.ndarray,
                "usa_r64": np.ndarray,
                "sst": np.ndarray,
                "usa_status": list,
                "dist2land": np.ndarray,
                "sid": str,
                "name": str,
                "season": int}

    # Required keys and basic types
    for key, expected_type in required.items():
        if key not in tc:
            errors.append(f"Missing required key: '{key}'")
            continue

        if not isinstance(tc[key], expected_type):
            errors.append(f"'{key}' must be {expected_type.__name__}, "
                          f"got {type(tc[key]).__name__}")

    # Stop here if basic structure is already invalid
    if errors:
        message = ("Invalid TC structure:\n\n"
                   + "\n".join(f"- {error}" for error in errors)
                   + "\n\n"
                   + TC_STRUCTURE)
        if raise_on_error:
            raise ValueError(message)
        return False

    # 1D arrays
    one_dimensional = ["davs",
                       "age",
                       "usa_wind",
                       "sst",
                       "dist2land"]

    for key in one_dimensional:
        if tc[key].ndim != 1:
            errors.append(f"'{key}' must be 1D, got shape {tc[key].shape}")

    if tc["dav_radii"].ndim != 3:
        errors.append(f"'dav_radii' must be 3D, got shape {tc['dav_radii'].shape}")

    elif tc["dav_radii"].shape[1] != 4:
        errors.append(f"'dav_radii' must have 4 quadrants, "
                      f"got shape {tc['dav_radii'].shape}")

    # Radius arrays
    for key in ("usa_r34", "usa_r50", "usa_r64"):
        if tc[key].ndim != 2:
            errors.append(f"'{key}' must be 2D, got shape {tc[key].shape}")
        elif tc[key].shape[1] != 4:
            errors.append(f"'{key}' must have 4 quadrants, "
                          f"got shape {tc[key].shape}")

    # Check that all time-dependent arrays have the same number
    # of observations.
    expected_length = len(tc["davs"])

    # usa_status is a list, so just validate its length
    if len(tc["usa_status"]) != expected_length:
        errors.append(f"'usa_status' has length {len(tc['usa_status'])}, "
                      f"expected {expected_length}")

    # DAV radial profile
    for key in ("dav_radii",
                "age",
                "usa_wind",
                "usa_r34",
                "usa_r50",
                "usa_r64",
                "sst",
                "dist2land"):
        if tc[key].shape[0] != expected_length:
            errors.append(f"'{key}' has length {tc[key].shape[0]}, "
                          f"expected {expected_length}")

    # Basin
    if tc["basin"] not in BASINS:
        errors.append(f"Unknown basin '{tc['basin']}'. "
                      f"Expected one of {BASINS}")

    if errors:
        message = ("Invalid TC structure:\n\n"
                   + "\n".join(f"- {error}" for error in errors)
                   + "\n\n"
                   + TC_STRUCTURE)
        if raise_on_error:
            raise ValueError(message)
        return False

    return True


class DAVRadiusTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=2000.0, step_km=PROFILE_RESOLUTION):
        self.threshold = threshold
        self.step_km = step_km

    def fit(self, X, y=None):
        # Automatically detect DAV columns in correct order
        self.dav_cols_ = sorted(
            [col for col in X.columns if col.startswith("dav_")],
            key=lambda x: int(x.split("_")[1].replace("km", ""))
        )
        return self

    def transform(self, X):
        dav = X[self.dav_cols_].values  # shape (N, K)

        # Condition: below threshold
        below = dav < self.threshold  # boolean array

        # Enforce "connected from first index"
        # cumulative product keeps True until first False
        connected = below.cumprod(axis=1)

        # Count how far we stay below threshold
        counts = connected.sum(axis=1)

        # Convert index → km
        radii = counts * self.step_km

        # Enforce: must start below threshold
        valid_start = below[:, 0]
        radii[~valid_start] = 0

        return radii.reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.array(["dav_radius"])


def _check_data_length_integrity(data_dir):
    """Double checks that the stored data has equal lengths for relevant parameters."""
    data = Path(data_dir)
    for tc_fn in data.glob(r"*/*/*.pkl"):
        with open(tc_fn, "rb") as file:
            tc = pickle.load(file)
        lengths = map(len, (tc['sst'],
                            tc['dav_radii'],
                            tc['age'],
                            tc['usa_wind'],
                            tc['usa_r34'],
                            tc['usa_r50'],
                            tc['usa_r64']))

        equal_length = len(set(lengths)) == 1
        if not equal_length:
            print("NOT EQUAL:", tc['name'], tc['season'], lengths)


def is_cyclonic(status):
    return np.isin(status, ("TS", "TY", "ST", "TC", "HU", "HR"))


def guard_RAM():
    """Check if too much RAM is likely to be used to loading process.

    Raises MemoryError if so."""
    mem = psutil.virtual_memory()
    sys_mem_available = mem.available / 1e9
    process_using = psutil.Process(os.getpid()).memory_info().rss / 1e9
    if sys_mem_available < MEMORY_SAFETY_FACTOR*process_using:  # GB limit
        raise MemoryError("Too much RAM used (won't be able to finish).")


def load_data(data_dir: str):
    """Loads entire dataset and puts it into a dictionary of dataframes.

    Each dataframe will be used to train it's own model. """
    data_lists = {basin: {quadrant: {radius: []
                                     for radius in RADII}
                          for quadrant in QUADRANTS}
                  for basin in BASINS}
    data = Path(data_dir)
    for tc_fn in data.glob(r"*/*/*.pkl"):
        guard_RAM()

        with open(tc_fn, "rb") as file:
            tc = pickle.load(file)
        validate_tc_structure(tc)

        basin = tc['basin']
        dav = tc["davs"]
        profile = tc["dav_radii"]
        tc_age = tc["age"]
        tc_wind = tc['usa_wind']

        observed = np.isfinite(dav) & ~np.isclose(dav, DEFAULT_DAV, atol=0.5)
        profile[~observed] = np.nan

        profile = smooth_dav(profile)
        tc_sst = forward_fill_sst(tc["sst"]) - 273  # Kelvin to Celcius

        useful = (observed
                  & np.isfinite(tc_age)
                  & np.isfinite(tc_sst)
                  & np.isfinite(tc_wind)
                  & is_cyclonic(tc["usa_status"])
                  & (tc["dist2land"] >= 300))

        for quadrant, radius in product(QUADRANTS, RADII):
            tc_radii = tc[RADII[radius]] * NM_KM

            if quadrant == "symmetric":
                quadrant_profile = quadrants_to_symmetrical(profile)
                tc_radii = quadrants_to_symmetrical(np.nan_to_num(tc_radii, nan=0.0))
                nonzero_radii = tc_radii != 0
            else:
                quadrant_profile = profile[:, QUADRANT_INDEX[quadrant]]
                tc_radii = tc_radii[:, QUADRANT_INDEX[quadrant]]
                nonzero_radii = ~np.isnan(tc_radii) & (tc_radii != 0)

            useful_radius = useful & nonzero_radii  # A huge amount of predictions should be 0. We can use a seperate model to determine if 0.

            profiles = quadrant_profile[useful_radius]  # (N, 75)
            df = pd.DataFrame(profiles, columns=[f"dav_{i*PROFILE_RESOLUTION}km" for i in range(profiles.shape[1])])

            df["age"] = tc_age[useful_radius]
            df["sst"] = tc_sst[useful_radius]
            df["radius"] = tc_radii[useful_radius]
            df["wind"] = tc_wind[useful_radius]
            df["storm_id"] = tc["sid"]

            data_lists[basin][quadrant][radius].append(df)

    output = {basin: {quadrant: {radius: pd.concat(data_lists[basin][quadrant][radius], ignore_index=True)
                                 for radius in RADII}
                      for quadrant in QUADRANTS}
              for basin in BASINS}

    return output


def _get_ridge_coefficients(results):
    best_model = results["Details"].best_estimator_

    # Get feature names after polynomial expansion
    feature_names = best_model[:-1].get_feature_names_out()

    # Get Ridge coefficients
    coefficients = best_model[-1].coef_

    importance = pd.DataFrame({"feature": feature_names,
                               "coefficient": coefficients,
                               "abs_coefficient": np.abs(coefficients)})

    importance = importance.sort_values("abs_coefficient",
                                        ascending=False)

    return importance


def _get_lasso_coefficients(results):
    best_model = results["Details"].best_estimator_

    feature_names = (best_model.named_steps["prep"].get_feature_names_out())
    poly_names = (best_model.named_steps["poly"].get_feature_names_out(feature_names))
    coef = best_model.named_steps["model"].coef_

    lasso_features = pd.DataFrame({"feature": poly_names,
                                   "coefficient": coef})
    lasso_features = (lasso_features.loc[lasso_features["coefficient"].abs() > 1e-10].sort_values("coefficient", key=abs, ascending=False))
    return lasso_features


def build_features(data, predicting_features):
    dav_cols = [c for c in data.columns if c.startswith("dav_")]

    use_dav = "dav_radii" in predicting_features

    X_parts = []
    transformers = []

    if use_dav:
        X_parts.extend(dav_cols)
        transformers.append(("dav_radius", DAVRadiusTransformer(), dav_cols))

    numerical_features = [f for f in predicting_features if f != "dav_radii"]

    if numerical_features:
        X_parts.extend(numerical_features)
        transformers.append(("num", "passthrough", numerical_features))

    X = data[X_parts]
    return X, transformers, use_dav, dav_cols


def build_pipeline(transformers):
    """Build the sci-kit learn pipeline."""
    preprocessor = ColumnTransformer(transformers)

    pipe = Pipeline([("prep", preprocessor),
                     ("poly", PolynomialFeatures(include_bias=False)),
                     ("scaler", StandardScaler()),
                     ("model", Ridge())])

    return pipe


def build_param_grid(models, alphas, poly_degrees, use_dav, thresholds):
    """Construct the grid that will be used to search for hyperparameters."""
    param_grid = []

    for model in models:
        base = {"model": [model],
                "model__alpha": alphas,
                "poly__degree": poly_degrees}

        if use_dav:
            base["prep__dav_radius__threshold"] = thresholds

        param_grid.append(base)

    return param_grid


def evaluate_model(grid, X_train, X_test, y_train, y_test):
    """Calculates metrics used to evaluate model performance."""
    best_model = grid.best_estimator_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mape = 100*mean_absolute_percentage_error(y_train, y_train_pred)
    test_mape = 100*mean_absolute_percentage_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Dummy baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_dummy_test = dummy.predict(X_test)
    dummy_rmse = np.sqrt(mean_squared_error(y_test, y_dummy_test))

    skill = 1 - (test_rmse / dummy_rmse)

    return {"Best Model - Train RMSE": train_rmse,
            "Best Model - Test RMSE": test_rmse,
            "Best Model - Train MAPE": train_mape,
            "Best Model - Test MAPE": test_mape,
            "Best Model - Train MAE": train_mae,
            "Best Model - Test MAE": test_mae,
            "Best Model - R2 Score": r2_score(y_test, y_test_pred),
            "Dummy Mean RMSE": dummy_rmse,
            "Skill Score": skill}


def fit(data: pd.DataFrame,
        poly_degrees=None,
        alphas=None,
        thresholds=None,
        random_seed=0,
        predicting_features=None,
        models=None):
    """Searches hyperparameters to fit a model to the data."""

    if poly_degrees is None:
        poly_degrees = [1, 2, 3]
    if alphas is None:
        alphas = [0.1, 1.0, 10.0]
    if thresholds is None:
        thresholds = [1500, 2000, 2500]
    if predicting_features is None:
        predicting_features = ["dav_radii", "age", "sst", "wind"]
    if models is None:
        models = [Ridge(), Lasso(max_iter=10000)]

    if not isinstance(data, pd.DataFrame):
        raise AttributeError(f"Data must be a dataframe. Got {type(data)}.")

    # --- Feature construction ---
    X, transformers, use_dav, dav_cols = build_features(data, predicting_features)
    y = data["radius"]
    groups = data["storm_id"]

    # --- Pipeline ---
    pipe = build_pipeline(transformers)

    # --- Param grid ---
    param_grid = build_param_grid(models, alphas, poly_degrees, use_dav, thresholds)

    # --- Train/test split ---
    splitter = GroupShuffleSplit(test_size=0.2, random_state=random_seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    grid = GridSearchCV(pipe,
                        param_grid,
                        cv=GroupKFold(n_splits=5),
                        scoring="neg_mean_squared_error",
                        n_jobs=-1)

    grid.fit(X_train, y_train, groups=groups_train)

    # --- Evaluation ---
    metrics = evaluate_model(grid, X_train, X_test, y_train, y_test)

    return {"Details": grid,
            **metrics}


def pipeline_to_json(results, variable_metadata, tol=1e-10, **fit_details):
    """Turn a model found to a JSON structure.

    Standardised parameters are returned to original units.
    """
    # --- Extract ---
    details = results['Details']
    rmse_train = results["Best Model - Train RMSE"]
    rmse_test = results["Best Model - Test RMSE"]
    mape_train = results["Best Model - Train MAPE"]
    mape_test = results["Best Model - Test MAPE"]
    mae_train = results["Best Model - Train MAE"]
    mae_test = results["Best Model - Test MAE"]
    pipe = details.best_estimator_

    dav_radius_threshold = details.best_params_.get("prep__dav_radius__threshold",
                                                    None)

    prep = pipe.named_steps["prep"]
    poly = pipe.named_steps["poly"]
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]

    # --- Feature names (base variables) ---
    base_feature_names = prep.get_feature_names_out()
    variables = [v.split("__")[-1] for v in base_feature_names]

    # --- Undo scaling ---
    coefs_scaled = model.coef_
    intercept_scaled = model.intercept_

    scale = scaler.scale_
    mean = scaler.mean_

    coefs = coefs_scaled / scale
    intercept = intercept_scaled - np.sum(coefs_scaled * mean / scale)

    # --- Get exact polynomial powers ---
    powers_matrix = poly.powers_

    # --- Build terms ---
    terms = []
    for coef, powers in zip(coefs, powers_matrix):
        if abs(coef) < tol:
            continue

        terms.append({"coef": float(coef),
                      "powers": powers.tolist()})

    metrics = {"rmse_train": float(rmse_train),
               "rmse_test": float(rmse_test),
               "mape_train": float(mape_train),
               "mape_test": float(mape_test),
               "mae_train": float(mae_train),
               "mae_test": float(mae_test)}

    # --- Final JSON ---
    model_json = {**fit_details,
                  "time_created": datetime.now(UTC).isoformat(),
                  "metrics": metrics,
                  "variables": variables,
                  "variable_metadata": variable_metadata,
                  "dav_radius_threshold": (float(dav_radius_threshold)
                                           if dav_radius_threshold is not None
                                           else None),
                  "intercept": float(intercept),
                  "terms": terms}

    return model_json


def fit_with_search(save_dir: str,
                    data_dir: str,
                    poly_degrees: list,
                    alphas: list,
                    thresholds: list,
                    random_seed: int,
                    models: list) -> None:
    """
    Creates the fit jsons for the basins, quadrants and radii. Searches optimal configs.

    Parameters
    ----------
    save_dir : str
        Where the jsons are to be saved to.
    data_dir : str
        Where the data is found.
    poly_degrees : list
        Which degrees to be searched for poly terms.
    alphas : list
        Regression alpha coefficient for training.
    thresholds : list
        Which DAV profile thresholds are to be searched
    random_seed : int
        Seed used to reproducibly partition storms into training and test sets.
        Grouped cross-validation is then performed on the training storms during
        hyperparameter selection.
    models : list
        Which models to use. Currently supports Ridge() or Lasso(max_iter=n) only.

    Creates JSONs for each fit.

    """
    data = load_data(data_dir)

    for basin, quadrant, radius in tqdm_product(BASINS, QUADRANTS, RADII):
        fit_details = {"basin": basin,
                       "quadrant": quadrant,
                       "radius": radius,
                       "profile_resolution": PROFILE_RESOLUTION}
        results = fit(data[basin][quadrant][radius],
                      poly_degrees=poly_degrees,
                      alphas=alphas,
                      thresholds=thresholds,
                      random_seed=random_seed,
                      models=models)

        model_json = pipeline_to_json(results, VARIABLE_METADATA, **fit_details)
        with open(Path(save_dir) /
                  f"MODEL_{basin}_{quadrant}_{radius}.json", "w") as f:
            json.dump(model_json, f, indent=4)


# if __name__ == "__main__":
#     save_dir = ""
#     data_dir = ""
#     fit_with_search(save_dir=save_dir,
#                     data_dir=data_dir,
#                     poly_degrees=[1, 2, 3],
#                     alphas=[1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.],
#                     thresholds=range(1500, 2801, 50),
#                     random_seed=0,
#                     models=[Ridge()])
