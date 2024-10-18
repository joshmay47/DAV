# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:34:50 2023

@author: Josh
"""

import numpy as np
from numba import cuda, njit
from functools import lru_cache
import math
import scipy
import pickle

PI = math.pi

class Updater:
    def __init__(self,total_length, update_frequency=0.01):
        self.proportion = 1/total_length
        self.current = 0
        if update_frequency is None:
            self.update_frequency = 0.1*math.ceil(10/total_length) #usually 0.1, every 10%
        else:
            self.update_frequency = update_frequency
        self.next_update = self.update_frequency

    def update(self):
        self.current += self.proportion
        if self.current >= self.next_update:
            print(f"{round(self.next_update*100)}% complete")
            self.next_update += self.update_frequency

@njit
def numba_arctan(dy,dx):
    return (180/PI*np.arctan2(dy,-dx))%360

def gradient_of_image(image):
    """

    Parameters
    ----------
    image : (n,n) numpy array, type float
        numpy array of nxn pixels of a satellite image, where pixels represent
        upwelling infrared temperature.

    Returns
    -------
    (n,n) numpy array, type float
        The direction, in degrees with range 0-360 of decreasing temperature,
        0 degrees is pointing to the east (positive x direction) with increasing
        angle going anti-clockwise

    """
    y_filt,x_filt = np.gradient(image)
    return numba_arctan(y_filt,x_filt)

@njit
def within_radius(x,y,radius):
    """

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
    return (x**2+y**2) < radius**2

def tan_with_limits(x,y):
    """

    Parameters
    ----------
    x : int or float
        distance from the point of origin in the x direction
    y : int or float
        distance from the point of origin in the y direction

    Raises
    ------
    ValueError
        Value error is raised when there is 0 distance in the x or y direction

    Returns
    -------
    float
        The direction from the point of origin, in degrees with range 0-360,
        0 degrees is pointing to the east (positive x direction) with increasing
        angle going anti-clockwise

    """
    if y == 0:
        if x==0:
            return 0
        return 270 if x < 0 else 90
    return (180/PI*math.atan2(x,y))%360

@lru_cache(maxsize=6)
def pixels_template(radius):
    """

    Parameters
    ----------
    radius : int or float
        radius of area analysed around the points of origin in pixels

    Returns
    -------
    template : (n,n) numpy array, boolean
        The pixels around the pixel of interest that will be involved in the
        calculation, will look like a circle if plotted.

    """
    size = int(2*radius)
    pixel_locs = np.zeros((size,size),dtype = bool)
    ray_angles = np.empty((size,size),dtype = np.float64)
    low = math.floor(radius)
    high = math.ceil(radius)
    for row in range(high+low):
        for col in range(high+low):
            ray_angles[row,col] = tan_with_limits(low-row,col-low)
            pixel_locs[row,col] = within_radius(row-low,col-low,radius)
    return pixel_locs,ray_angles

@cuda.jit("f4(f4)", device=True)
def rectify_angle(val): #Prof changed offsets to float
    if val < -90:
        if val < -270:
            return val+np.float32(360)
        return -val-np.float32(180)
    if val > 90:
        if val > 270:
            return val-np.float32(360)
        return np.float32(180)-val
    return val

@cuda.jit(device=True)
def individual_dav(gradient, center_pixel, mask, mask_angles):
    """
    Calculate the variance of the surrounding pixels around a given center pixel using a mask on the GPU.

    Parameters:
    - image: The input image as a NumPy array.
    - center_pixel: Tuple (x, y) representing the coordinates of the center pixel.
    - mask: NumPy boolean array specifying the pixels to include (True) and exclude (False).
    - variance_result: Output array to store the variance.

    Returns: float32, DAV value for the centre_pixel
    """
    mask_radius = len(mask)//2
    center_x, center_y = center_pixel

    # Calculate the bounds for slicing the image, ensuring they are within image boundaries
    start_x = max(center_x - mask_radius, 0)
    mask_offset_x = max(mask_radius-center_x, 0)
    end_x = min(center_x + mask_radius, gradient.shape[1])
    start_y = max(center_y - mask_radius, 0)
    mask_offset_y = max(mask_radius-center_y, 0)
    end_y = min(center_y + mask_radius, gradient.shape[0])
    image_to_mask_y = - start_y + mask_offset_y
    image_to_mask_x = - start_x + mask_offset_x

    # Initialize variables for one-pass variance calculation
    count = 0
    p1 = np.float32(0)  # Sum of pixel values
    p2 = np.float32(0)  # Sum of squared pixel values

    # Iterate over the pixels within the mask
    for i in range(start_y, end_y):
        for j in range(start_x, end_x):
            mask_y, mask_x = i + image_to_mask_y, j + image_to_mask_x
            if mask[mask_y, mask_x]:
                value = rectify_angle(gradient[i, j]-mask_angles[mask_y, mask_x])
                p1 += value
                p2 += value * value
                count += 1

    # Store the result in the output array
    return (p2-p1*p1/count)/(count - 1)

def fill(image):
    mask = np.isnan(image)
    image[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), image[~mask])
    return image

def blur(image, filling = True):
    if filling:
        image = fill(image)
    return scipy.ndimage.gaussian_filter(image,sigma=2)

@cuda.jit
def dav_kernel(d_gradient,d_skip_mask,d_mask,d_mask_angles,d_out):
    x,y = cuda.grid(2)
    if y < d_gradient.shape[0] and x < d_gradient.shape[1]:
        if d_skip_mask[y,x]:
            d_out[y,x] = np.nan
        else:
            d_out[y,x] = individual_dav(d_gradient,(x,y),d_mask,d_mask_angles)
        
@cuda.jit
def centre_dav_kernel(d_gradients,d_mask,d_mask_angles,d_out):
    z,y,x = cuda.grid(3)
    low = d_gradients.shape[1]//2-3
    high = low+5
    if (z < d_gradients.shape[0]) and (low < y < high) and (low < x < high):
        d_out[z,y-low,x-low] = individual_dav(d_gradients[z],(x,y),d_mask,d_mask_angles)

def dav(images,radius,skip_masks = None,verbose=False):
    """

    Parameters
    ----------
    images : 2D or 3D numpy array, of shape NxHxW or HxW
        Brightness temperature image(s), must be lat-lon projection and float
    radius : int or float
        The radius of the DAV operation, specified in pixels, not km or degrees.
    skip_masks : 2D or 3D numpy bool_ array, optional
        If specified, must be the same shape as images. When the pixel is true,
        the DAV operation is not calculated for the corresponding image pixel. 
        Can be used to speed up the operation when some areas are not desired.
    verbose : bool, optional
        If true, provides updates on the progress to the console. The default is False.

    Returns
    -------
    2D or 3D numpy array, of shape NxHxW or HxW
        The full DAV map corresponding to the images parameter. Will be of the
        same shape and type.

    """
    if skip_masks is None:
        skip_masks = np.zeros_like(images,dtype=np.bool_)
    if len(images.shape) == 2:
        images = np.expand_dims(images,axis=0)
        skip_masks = np.expand_dims(skip_masks,axis=0)
        contract_final_dims = True
    else:
        contract_final_dims = False
    assert images.shape == skip_masks.shape, f"images and skip_masks should have the same shape, got {images.shape} and {skip_masks.shape}, respectively."
    
    x_size = images.shape[2]
    y_size = images.shape[1]
    dav_images = np.empty_like(images,dtype=np.float32)
    template,ray_angles = pixels_template(radius)
    
    d_template = cuda.to_device(template)
    d_ray_angles = cuda.to_device(ray_angles.astype(np.float32))
    d_results = cuda.device_array((y_size, x_size), dtype=np.float32)
    tpb = (8,8)
    bpg = ((x_size + 7) // 8,(y_size + 7) // 8)
    
    if verbose: tracker = Updater(images.shape[0])
    if verbose: print("Beginning DAV:")
    for i,(image,skip_mask) in enumerate(zip(images,skip_masks)):
        gradient = gradient_of_image(blur(image)).astype(np.float32)
        d_skip_mask = cuda.to_device(skip_mask)
        d_gradient = cuda.to_device(gradient)
        dav_kernel[bpg,tpb](d_gradient,d_skip_mask,d_template,d_ray_angles,d_results)
        dav_result = d_results.copy_to_host()
        dav_images[i] = dav_result
        if verbose: tracker.update()
    if contract_final_dims:
        return np.squeeze(dav_images,axis=0)
    return dav_images

def centre_davs(images, radius):
    """

    Parameters
    ----------
    images : 3D numpy array, of shape NxHxW
        Brightness temperature image(s), must be lat-lon projection and float
    radius : int or float
        The radius of the DAV operation, specified in pixels, not km or degrees.

    Returns
    -------
    1D numpy array of shape N,
        The DAV values in the centre of the images.

    """
    centre_x, centre_y = images.shape[2]//2, images.shape[1]//2
    cut_amt = radius+3
    images = np.array(images[:,
                             centre_y-cut_amt:centre_y+cut_amt,
                             centre_x-cut_amt:centre_x+cut_amt],copy=True)
    dav_pixels = np.zeros((images.shape[0],4,4), dtype=np.float32)
    template, ray_angles = pixels_template(radius)
    gradients = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images): gradients[i] = gradient_of_image(blur(image))
    
    d_template = cuda.to_device(template)
    d_ray_angles = cuda.to_device(ray_angles)
    d_results = cuda.to_device(dav_pixels)
    d_gradients = cuda.to_device(gradients)
    tpb = (4,4,4)
    bpg = (math.ceil(gradients.shape[0]/4),
           math.ceil(gradients.shape[1]/4),
           math.ceil(gradients.shape[2]/4))
    
    centre_dav_kernel[bpg,tpb](d_gradients, d_template, d_ray_angles, d_results)
    
    return np.mean(d_results.copy_to_host(),axis=(1,2))
