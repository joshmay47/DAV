"""
Preprocessing utilities for tropical cyclone DAV data.

This module provides utilities for preparing DAV and sea surface temperature
data for tropical cyclone wind-radius modelling.

The primary public functions are:

* :func:`smooth_dav` -- applies NaN-aware temporal smoothing to DAV profiles.
* :func:`forward_fill_sst` -- fills missing sea surface temperature values
  using the most recent preceding observation.

Other functions in this module support these operations and are implementation
details rather than part of the intended public API.

Dependencies
------------
numpy
scipy

Notes
-----
The temporal smoothing performed by :func:`smooth_dav` uses a fixed-width
window and handles missing values without allowing NaNs to contaminate
neighbouring valid observations.

:func:`forward_fill_sst` performs causal filling: a missing SST observation is
replaced only by an earlier valid observation and never by a future
observation.

Author
------
Joshua May <josh.w.may@gmail.com>
"""

import numpy as np
from scipy import ndimage

SMOOTHING_RECORDS = 48  # In half-hours


def smooth_dav(profile, distance=SMOOTHING_RECORDS):
    """NaN-aware, fixed temporal smoothing along each radial profile."""
    valid = np.isfinite(profile)
    kernel = np.ones(distance)
    numerator = ndimage.convolve1d(np.where(valid, profile, 0.0), kernel, axis=0, mode="reflect")
    denominator = ndimage.convolve1d(valid.astype(float), kernel, axis=0, mode="reflect")
    return np.divide(numerator, denominator, out=np.full_like(profile, np.nan), where=denominator > 0)


def forward_fill_sst(sst):
    """Causal fill: missing SST never uses a later observation."""
    sst = np.asarray(sst, dtype=float).copy()
    first = np.flatnonzero(np.isfinite(sst))

    if not len(first):
        return sst

    sst[:first[0]] = np.nan
    for i in range(first[0] + 1, len(sst)):
        if not np.isfinite(sst[i]):
            sst[i] = sst[i - 1]

    return sst


def quadrants_to_symmetrical(array):
    return np.mean(array, axis=1)
