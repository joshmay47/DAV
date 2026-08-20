# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:23:24 2024

@author: Josh
"""
from functools import lru_cache
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DAVProfile:
    values: np.ndarray
    resolution: float


def bilinear_indexing(data, x_indicies, y_indicies):
    """Indexes x_indices and y_indices in data using bilinear interpolation."""
    assert y_indicies.shape == x_indicies.shape, \
        "x_indicies and y_indicies must have the same shape"
    left   = np.floor(x_indicies).astype(np.int64)
    right  = left + 1
    top    = np.floor(y_indicies).astype(np.int64)
    bottom = top + 1

    left   = np.clip(left,   0, data.shape[1]-1)
    right  = np.clip(right,  0, data.shape[1]-1)
    top    = np.clip(top,    0, data.shape[0]-1)
    bottom = np.clip(bottom, 0, data.shape[0]-1)

    left_dist   = x_indicies - left
    top_dist    = y_indicies - top
    right_dist  = right  - x_indicies
    bottom_dist = bottom - y_indicies

    tl_contribution = right_dist * bottom_dist * data[top,    left]
    bl_contribution = right_dist * top_dist    * data[bottom, left]
    tr_contribution = left_dist  * bottom_dist * data[top,    right]
    br_contribution = left_dist  * top_dist    * data[bottom, right]

    return tl_contribution + bl_contribution + tr_contribution + br_contribution


def _handle_polar_inputs(image_shape, centre, radius_range, angle_linspace):
    max_y, max_x = image_shape
    if centre is None:
        cy, cx = max_y/2, max_x/2
    else:
        cy, cx = centre
        assert 0 <= cy <= max_y
        assert 0 <= cx <= max_x

    max_radius_possible = min(cy, cx, max_y-cy, max_x-cx)
    if radius_range is None:
        radius_range = (1, max_radius_possible, 1)
    else:
        assert radius_range[0] < radius_range[1] <= max_radius_possible

    if angle_linspace is None:
        angle_linspace = (0, 2*np.pi, 360)
    else:
        assert 0 <= angle_linspace[0] < angle_linspace[1] <= 2*np.pi

    return (cy, cx), radius_range, angle_linspace


@lru_cache(maxsize=10)
def _get_polar_indicies(image_shape, centre, radius_range, angle_linspace):
    centre, radius_range, angle_linspace = _handle_polar_inputs(image_shape,
                                                                centre,
                                                                radius_range,
                                                                angle_linspace)
    radii = np.expand_dims(np.arange(*radius_range), axis=1)
    angles = np.expand_dims(np.linspace(*angle_linspace), axis=0)
    y_indicies = -radii*np.cos(angles)+centre[0]
    x_indicies = radii*np.sin(angles)+centre[1]
    return y_indicies, x_indicies


def to_polar(image,
             centre=None,
             radius_range=None,
             angle_linspace=None,
             method='bilinear') -> np.ndarray:
    """Turn an image into a polar view."""
    y_indicies, x_indicies = _get_polar_indicies(image.shape, centre,
                                                 radius_range, angle_linspace)
    if method == 'nearest':
        return image[y_indicies.astype(int), x_indicies.astype(int)]
    if method == 'bilinear':
        return bilinear_indexing(image, x_indicies, y_indicies)
    raise ValueError(f"method should be 'bilinear' or 'nearest', not '{method}'.")


def down_angle(image: np.ndarray,
               scaling_factor: int):
    """Reduces the number of values in the 2nd dimension by taking the mean"""
    result_width, remainder = divmod(image.shape[1], scaling_factor)
    if remainder != 0:
        raise ValueError(f"{scaling_factor=} must be an integer multiple of {image.shape[1]=}")
    a = image.reshape(-1, scaling_factor).mean(axis=1)
    return a.reshape(image.shape[0], result_width)


def _get_fft(image, order=8):
    IMG = np.fft.fft(image)*2*order/image.shape[1]
    return np.concatenate((IMG[:,:order], IMG[:,1-order:]), axis=1)


def _to_data(dav_image, order=8):
    polar_image = to_polar(dav_image, radius_range=(1,50))
    fft_image = _get_fft(polar_image, order)
    lower_polar_image = down_angle(polar_image, 24)
    data = np.concatenate((lower_polar_image.flatten(), fft_image.real.flatten(), fft_image.imag.flatten()))
    return data


def get_dav_profile(dav_images: np.ndarray,
                    resolution: float,
                    radius: int = 75) -> np.ndarray:
    """
    Generate a DAV profile from an array of DAV maps.

    Parameters
    ----------
    dav_images : np.ndarray
        Images of a tropical cyclone after running the DAV operation on them.
        Should have three dimensions (time, height, width).
    resolution : float
        Resolution of the input dav_images in km/pixel.
    radius : int
        Number of pixels to create the profile away from the center of the
        image. Default is 75 (for a 8km/pixel resolution, this is 600km).

    Returns
    -------
    profile : np.ndarray
        A (time, 4, 75) shaped np.array where the values correspond to
        (time, [ne, se, sw, nw], [radius - up to 75 pixels]).

    """
    if not isinstance(dav_images, np.ndarray):
        raise TypeError("Expected dav_images to be of type np.ndarray, got "
                        f"{type(dav_images)}.")

    if dav_images.ndim != 3:
        raise ValueError("Unexpected dimensionality of DAV array, should be "
                         f"3D, got {dav_images.ndim}D.")
    views = [to_polar(image, radius_range=(0, 75)) for image in dav_images]
    profile = np.array([down_angle(image, 90).T for image in views])
    return DAVProfile(values=profile, resolution=resolution)
