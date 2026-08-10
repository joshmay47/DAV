"""
Deviation Angle Variance (DAV) computation utilities.

This module provides GPU-accelerated functions for computing DAV from
satellite infrared imagery, including full-image maps and central estimates.

Originally concieved in: M. F. Pineros, E. A. Ritchie and J. S. Tyo,
"Objective Measures of Tropical Cyclone Structure and Intensity Change From
Remotely Sensed Infrared Image Data" doi: 10.1109/TGRS.2008.2000819.

Example:
-------
        from dav.generate import dav, centre_dav

        dav_radius_km = 300 # radius of dav calculation in km
        image_resolution = 2.22 # km per pixel
        radius_pixels = dav_radius_km/image_resolution

        images = load_images() # placeholder function for loading images

        dav_maps = dav(images, radius_pixels)
        cdavs = centre_dav(images, radius_pixels)

Dependencies:
-------------
    numpy
    numba
    scipy

Author: Joshua May (josh.w.may@gmail.com)

"""

import numpy as np
from numba import cuda, njit
from functools import lru_cache
import math
import scipy

from scipy.signal import convolve2d

PI = math.pi

sobellike_v = np.array([[-1, -1,  0, -1, -1],
                        [-1, -1,  0, -1, -1],
                        [0,   0,  0,  0,  0],
                        [1,   1,  0,  1,  1],
                        [1,   1,  0,  1,  1]])

sobellike_h = sobellike_v.T


@njit
def numba_arctan(dy, dx):
    """
    Arctan function implemented with numba.

    Parameters
    ----------
    dy : int or float
        distance from the point of origin in the y direction
    dx : int or float
        distance from the point of origin in the x direction


    Returns
    -------
    float
        The direction from the point of origin, in degrees with range 0-360,
        0 degrees is pointing to the east (positive x direction) with
        increasing angle going anti-clockwise

    """
    return (180/PI*np.arctan2(dy, dx)) % 360


def gradient(im_obj):
    """
    Calculate vertical and horizontal gradient of an image.

    im_obj: array
        2D image of brightness temperatures.

    Returns
    -------
    G_v: array
        Gradient in vertical direction.
    G_h: array
        Gradient in horizontal direction.
    """
    G_v = convolve2d(im_obj, sobellike_v, mode='same')
    G_h = convolve2d(im_obj, sobellike_h, mode='same')
    return G_v, G_h


def gradient_of_image(image):
    """
    Calculate the gradient of an image in terms of angle anti-clockwise from east.

    Parameters
    ----------
    image : (n,n) numpy array, type float
        numpy array of nxn pixels of a satellite image, where pixels represent
        upwelling infrared temperature.

    Returns
    -------
    (n,n) numpy array, type float
        The direction, in degrees with range 0-360 of decreasing temperature,
        0 degrees is pointing to the east (positive x direction) with
        increasing angle going anti-clockwise

    """
    y_filt, x_filt = gradient(image)
    return numba_arctan(y_filt, x_filt)


@njit
def within_radius(x, y, radius):
    """
    Check if x and y distances are within a radius.

    Parameters
    ----------
    x : int or float
        distance from the point of origin in the x direction
    y : int or float
        distance from the point of origin in the y direction
    radius : int or float
        The maximum allowable distance from the point of origin in pixels

    Returns
    -------
    bool
        Whether the x and y combine to be within range (less than radius)

    """
    return (x**2+y**2) <= radius**2


@lru_cache(maxsize=6)
def pixels_template(radius):
    """
    Generate a template for calculating ray angles.

    Parameters
    ----------
    radius : int or float
        radius of area analysed around the points of origin in pixels

    Returns
    -------
    pixel_locs : (n,n) numpy array, boolean
        The pixels around the pixel of interest that will be involved in the
        calculation, will look like a circle if plotted.

    ray_angles : (n,n) numpy array, boolean
        The angles from the center of each pixel in pixel_locs.

    """
    size = int(2*radius+1)
    pixel_locs = np.zeros((size, size), dtype=bool)
    ray_angles = np.empty((size, size), dtype=np.float32)
    low = math.floor(radius)
    high = math.ceil(radius)+1
    for row in range(high+low):
        for col in range(high+low):
            ray_angles[row, col] = numba_arctan(row-low, col-low)
            pixel_locs[row, col] = within_radius(row-low, col-low, radius)
    return pixel_locs, ray_angles


@cuda.jit(device=True)
def deviation_angle(primary_angle, secondary_angle):
    """
    Calculate the angle deviation between two angles.

    Parameters
    ----------
    primary_angle : float
        The angle from the individual DAV pixel to the pixel that is being
        considered.
    secondary_angle : float
        The gradient of the pixel that is being considered.

    Returns
    -------
    difference: float
        The angle deviation of the secondary_angle from the primary_angle.

    """
    diff = secondary_angle - primary_angle
    if diff < -90.0:
        if diff < -270.0:
            return diff+360.0
        return diff+180.0
    if diff > 90.0:
        if diff > 270.0:
            return diff-360.0
        return diff-180.0
    return diff

@cuda.jit(device=True)
def _numba_min(a, b):
    return a if a > b else b

@cuda.jit(device=True)
def _numba_max(a, b):
    return a if a < b else b

@cuda.jit(device=True)
def individual_dav(gradient, center_x, center_y, mask, mask_angles):
    """
    Calculate the deviation angle variance for a single pixel on the GPU.

    Parameters
    ----------
    gradient : 2D array
        The input gradient image as a NumPy array.
    center_x : int
        Together with center_y, representing the coordinates of the center 
        pixel.
    center_y : int
        Together with center_x, representing the coordinates of the center 
        pixel.
    mask : array
        NumPy boolean array specifying the pixels to include (True) and
        exclude (False).
    mask_angles : array
        Angles in the mask array from the center pixel, should be the same
        shape.

    Returns
    -------
    float32
        DAV value for the centre_pixel

    """
    mask_radius = mask.shape[0]//2

    # Calculate the bounds for slicing the image, ensuring they are within
    # image boundaries
    # X bounds
    start_x = _numba_min(center_x - mask_radius, 0)
    mask_offset_x = _numba_min(mask_radius - center_x, 0)
    end_x = _numba_max(center_x + mask_radius, gradient.shape[1])
    
    start_y = _numba_min(center_y - mask_radius, 0)
    mask_offset_y = _numba_min(mask_radius - center_y, 0)
    end_y = _numba_max(center_y + mask_radius, gradient.shape[0])

    image_to_mask_y = - start_y + mask_offset_y
    image_to_mask_x = - start_x + mask_offset_x

    # Initialize variables for one-pass variance calculation
    count = np.int32(0)
    p1 = np.float32(0.0)  # Sum of pixel values
    p2 = np.float32(0.0)  # Sum of squared pixel values

    # Iterate over the pixels within the mask
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            mask_y, mask_x = i + image_to_mask_y, j + image_to_mask_x
            if mask[mask_y, mask_x]:
                value = deviation_angle(mask_angles[mask_y, mask_x],
                                        gradient[i, j])
                p1 += value
                p2 += value * value
                count += 1

    # Return the Variance using formula (sum(x^2)-sum(x)^2/N)/(count-1)
    return (p2-p1*p1/count)/(count - 1)


def fill(image):
    """
    Fill in the NaN values in an image.

    Parameters
    ----------
    image : 2D array
        Image that has NaN values.

    Returns
    -------
    image : array
        Image where the NaN values have been filled in from the nearest valid
        pixels to the left and right of the missing pixel.

    """
    mask = np.isnan(image)
    missing_all = mask.all()
    if not missing_all:
        image[mask] = np.interp(np.flatnonzero(mask),
                                np.flatnonzero(~mask),
                                image[~mask])
    return image


def blur(image, filling=True):
    """
    Blur an image using a gaussian filter.

    Parameters
    ----------
    image : 2D array
        Image that needs to be blurred. Will be the brightness temperature
        image in practice.
    filling : bool, optional
        Whether the image should be filled in before blurring. The default is
        True.

    Returns
    -------
    image : 2D array
        A blurred copy of the input image.

    """
    if filling:
        image = fill(image)
    return scipy.ndimage.gaussian_filter(image, sigma=1)


@cuda.jit
def dav_kernel(d_gradient, d_skip_mask, d_mask, d_mask_angles, d_out):
    """
    Cuda kernel for calculating DAV values in an image.

    Parameters
    ----------
    d_gradient : cuda.DeviceNDArray
        Angles from east (increasing anti-clockwise) of decreasing brightness
        in an image, this must be on the GPU.
    d_skip_mask : cuda.DeviceNDArray
        Boolean array of whether the pixels should be skipped, pixels are
        skipped if pixel value here is True. Same shape as d_gradient, this
        must be on the GPU.
    d_mask : cuda.DeviceNDArray
        Whether a pixel surrounding an individual pixel is used to contribute
        to the angle variances. This is filtered by proximity (DAV operation
        radius parameter), this must be on the GPU.
    d_mask_angles : cuda.DeviceNDArray
        Angle from the center pixel of the d_mask pixels. Same shape as d_mask,
        this must be on the GPU.
    d_out : cuda.DeviceNDArray
        The output Output array populated with computed values. Same shape as
        d_gradient, this must be on the GPU.

    """
    x, y = cuda.grid(2)
    if y < d_gradient.shape[0] and x < d_gradient.shape[1]:
        if d_skip_mask[y, x]:
            d_out[y, x] = np.nan
        else:
            d_out[y, x] = individual_dav(d_gradient,
                                         x,
                                         y,
                                         d_mask,
                                         d_mask_angles)


@cuda.jit
def centre_dav_kernel(d_gradient, d_mask, d_mask_angles, d_out):
    """
    Variant of regular cuda kernel for calculating the central DAV values in an image.

    Parameters
    ----------
    d_gradient : cuda.DeviceNDArray
        Angles from east (increasing anti-clockwise) of decreasing brightness
        in an image, this must be on the GPU.
    d_mask : cuda.DeviceNDArray
        Whether a pixel surrounding an individual pixel is used to contribute
        to the angle variances. This is filtered by proximity (DAV operation
        radius parameter), this must be on the GPU.
    d_mask_angles : cuda.DeviceNDArray
        Angle from the center pixel of the d_mask pixels, this must be on the
        GPU.
    d_out : cuda.DeviceNDArray
        The output array, the values will get populated by running this
        function, this must be on the GPU.

    """
    z, y, x = cuda.grid(3)
    low = d_gradient.shape[1]//2-3
    high = low+5
    if (z < d_gradient.shape[0]) and (low < y < high) and (low < x < high):
        d_out[z, y-low, x-low] = individual_dav(d_gradient[z],
                                                x,
                                                y,
                                                d_mask,
                                                d_mask_angles)


def dav(images: np.array,
        radius: float,
        skip_masks: np.array = None,
        global_inputs: bool = False):
    """
    Compute Deviation Angle Variance (DAV) for image data.

    DAV measures the variance of angular deviations between local gradient
    directions and radial directions from each pixel. It is used to quantify
    structural organization in tropical cyclone infrared imagery.

    Parameters
    ----------
    images : 2D or 3D numpy array, of shape NxHxW or HxW
        Brightness temperature image(s), must be lat-lon projection and float
    radius : int or float
        The radius of the DAV operation, specified in pixels, NOT km or
        degrees.
    skip_masks : 2D or 3D numpy bool_ array, optional
        If specified, must be the same shape as images. When the pixel is true,
        the DAV operation is not calculated for the corresponding image pixel.
        Can be used to speed up the operation when some areas are not desired.
    global_inputs: bool, optional
        If true, treats the input as a global image, reducing low-sampling
        artefacts along the horizontal borders of the image(s).

    Returns
    -------
    2D or 3D numpy array, of shape NxHxW or HxW
        The full DAV map corresponding to the images parameter. Will be of the
        same shape and type.

    Notes
    -----
    - Requires a CUDA-capable GPU.
    - Computation cost scales with image size and radius. O(N*r^2) where N is
    number of pixels and r is radius.

    """
    if skip_masks is None:
        skip_masks = np.zeros_like(images, dtype=np.bool_)
    if global_inputs:
        return _global_davs(images, radius, skip_masks)
    if len(images.shape) == 2:
        images = np.expand_dims(images, axis=0)
        skip_masks = np.expand_dims(skip_masks, axis=0)
        contract_final_dims = True
    else:
        contract_final_dims = False
    good_masking = images.shape == skip_masks.shape
    assert good_masking, ("images and skip_masks should have the same shape, "
                          f"got {images.shape} and {skip_masks.shape}, "
                          "respectively.")

    x_size = images.shape[2]
    y_size = images.shape[1]
    dav_images = np.empty_like(images, dtype=np.float32)
    template, ray_angles = pixels_template(radius)

    d_template = cuda.to_device(template.astype(np.float32))
    d_ray_angles = cuda.to_device(ray_angles.astype(np.float32))
    d_results = cuda.device_array((y_size, x_size), dtype=np.float32)
    tpb = (8, 8)
    bpg = ((x_size + 7) // 8, (y_size + 7) // 8)

    for i, (image, skip_mask) in enumerate(zip(images, skip_masks)):
        gradient = gradient_of_image(blur(image)).astype(np.float32)
        d_skip_mask = cuda.to_device(skip_mask)
        d_gradient = cuda.to_device(gradient)
        dav_kernel[bpg, tpb](d_gradient,
                             d_skip_mask,
                             d_template,
                             d_ray_angles,
                             d_results)
        dav_result = d_results.copy_to_host()
        dav_images[i] = dav_result
    if contract_final_dims:
        return np.squeeze(dav_images, axis=0)
    return dav_images


def _global_davs(images, radius, skip_masks):
    """
    Add support for wrap-around for the values used in DAV.

    Extends the input images in the DAV operation so that the end and start
    data from the start and end, respectively - are used. This reduces
    artefacts in the DAV output along the left and right edges of the image.

    This should only be called from using the global_inputs=True argument from
    dav()
    """
    BLUR_KERNEL_RADIUS = 8
    radius_safe = int(np.ceil(radius))
    extending_distance = BLUR_KERNEL_RADIUS + radius_safe

    images = np.concatenate((images,
                             images[..., :2*extending_distance]), axis=-1)

    skip_masks_leftside = skip_masks[..., :extending_distance]
    skip_masks = np.concatenate((skip_masks,
                                 skip_masks_leftside,
                                 np.ones_like(skip_masks_leftside)
                                 ), axis=-1)
    skip_masks[..., :extending_distance] = True

    davs = dav(images, radius, skip_masks, global_inputs=False)
    return np.roll(davs[..., extending_distance:-extending_distance],
                   extending_distance,
                   axis=-1)


def centre_dav(images: np.array,
               radius: float):
    """
    Calculate the mean of central DAV 4x4 pixels value in a series of images.

    Parameters
    ----------
    images : 3D numpy array, of shape NxHxW
        Brightness temperature images, must be lat-lon projection and float.
    radius : int or float
        The radius of the DAV operation, specified in pixels, NOT km or
        degrees.

    Returns
    -------
    1D numpy array of shape N,
        The DAV values in the centre of the images.

    Notes
    -----
    - Uses a cropped region around the image center for efficiency.
    - Requires a CUDA-capable GPU.
    """
    centre_x, centre_y = images.shape[2]//2, images.shape[1]//2
    cut_amt = int(np.ceil(radius))+3
    images = np.array(images[:,
                             centre_y-cut_amt:centre_y+cut_amt,
                             centre_x-cut_amt:centre_x+cut_amt], copy=True)
    dav_pixels = np.zeros((images.shape[0], 4, 4), dtype=np.float32)
    template, ray_angles = pixels_template(radius)
    gradients = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images):
        gradients[i] = gradient_of_image(blur(image)).astype(np.float32)

    d_template = cuda.to_device(template)
    d_ray_angles = cuda.to_device(ray_angles)
    d_results = cuda.to_device(dav_pixels)
    d_gradients = cuda.to_device(gradients)
    tpb = (4, 4, 4)
    bpg = (math.ceil(gradients.shape[0]/4),
           math.ceil(gradients.shape[1]/4),
           math.ceil(gradients.shape[2]/4))

    centre_dav_kernel[bpg, tpb](d_gradients,
                                d_template,
                                d_ray_angles,
                                d_results)

    return np.mean(d_results.copy_to_host(), axis=(1, 2))
