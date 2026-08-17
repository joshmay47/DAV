"""
Utility to calculate the tc_age from wind data

Dependencies
------------
numpy

Notes
-----
tc_age is more accurate with a higher samples_per_hour

Author
------
Joshua May <josh.w.may@gmail.com>
"""

import numpy as np


def get_tc_age(wind: np.ndarray,
               samples_per_hour: float = 2) -> np.ndarray:
    """Get the age of the TC (hours since reaching 34 kt).

    wind : np.ndarray
        TC VMax/wind speed in kt
    samples_per_hour : float
        How many samples per hour for the wind values.

    The time gets reset every time the TC becomes non-cyclonic again.
    """
    cyclonic = wind >= 34
    falling_under_threshold = np.where(np.diff(cyclonic.astype(float)) == -1)[0]
    cumulated_cyclonic_time = np.cumsum(cyclonic)
    for time in falling_under_threshold:
        cumulated_cyclonic_time[time:] -= cumulated_cyclonic_time[time]
    return cumulated_cyclonic_time/samples_per_hour
