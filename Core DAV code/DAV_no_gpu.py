# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:02:42 2023

@author: Josh
"""

import numpy as np
import numba as nb
from scipy import ndimage

@nb.vectorize
def arctan(y,x):
    return 57.29578*np.arctan2(y,x)%360

def generate_mask(radius):
    points = np.arange(-round(radius)+0.5, round(radius)+0.5)
    y,x = np.meshgrid(points, points, indexing='ij',sparse=True)
    mask = y**2 + x**2 <= radius**2
    ray_angles = arctan(y,x) # For East = 0, north = 90, (-y,x)
    return mask, ray_angles

def gradient_of(image):
    blurred_image = ndimage.gaussian_filter(image, sigma=2, mode='nearest')
    dy = ndimage.sobel(blurred_image, axis=0)
    dx = ndimage.sobel(blurred_image, axis=1)
    gradient = arctan(dy,dx) # For East = 0, north = 90, (dy,-dx)
    return gradient # Might also want to return the magnitude of the gradient

def centre_of_images(images, radius):
    width = radius + 5
    ni, ny, nx = images.shape
    cy, cx = ny//2, nx//2
    return images[:,cy-width:cy+width,cx-width:cx+width]

@nb.vectorize
def rectify(angle):
    if angle < -90:
        if angle < -270:
            return angle+360
        return angle+180
    if angle > 90:
        if angle > 270:
            return angle-360
        return angle-180
    return angle

@nb.njit
def var(deviations):
    n = len(deviations)
    p1=p2=0.0
    for val in deviations:
        p1 += val
        p2 += val*val
    return (p2-p1*p1/n)/(n+1)

def cdavs(images, radius):
    mask, ray_angles = generate_mask(radius)
    ray_angles = ray_angles[mask]
    gradients = [gradient_of(image) for image in centre_of_images(images.astype(np.int32), radius)]
    gradients = np.stack(gradients)[:,5:-5,5:-5]
    deviations = rectify(gradients[:,mask]-ray_angles)
    return [var(deviation) for deviation in deviations]
