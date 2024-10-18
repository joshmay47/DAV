# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:45:08 2024

@author: Josh
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from math import cos, sin, radians
from typing import Tuple

class Image:
    def __init__(self, bti: np.ndarray, resolution: float = 4):
        assert len(bti.shape) == 2, f"Must be a 2D image, got {len(bti.shape)}D."
        assert resolution > 0, f"Resolution must be a positive number, got {resolution}."
        self.bti = bti
        self.resolution = resolution
    
    def create_plot(self):
        fig = plt.figure(figsize=(self.bti.shape[0]//100, self.bti.shape[1]//100), frameon=False)
        ax = plt.gca()
        ax.imshow(self.bti[::-1], cmap="gray_r", interpolation='none', vmin=73, vmax=223)
        ax.axis('off')
        plt.gcf().set_dpi(100)
        return fig, ax
    
    def show(self):
        fig, ax = self.create_plot()
        plt.show()
    
    @classmethod
    def _check_wind_radii(self, data):
        if len(data) != 4:
            raise ValueError(f"wind_radii must have exactly four quadrants, got {len(data)}.")
        for element, quadrant in zip(data, ("NE", "SE", "SW", "NW")):
            if not isinstance(element, (int, float)) or element < 0:
                raise ValueError(f"All quadrants must have positive floats, quadrant: {quadrant}, was {element}.")
            
    
    def show_wind_radii(self, wind_radii: Tuple[float, float, float, float], fn):
        """
        wind_radii: tuple of length 4, of the form: (NE, SE, SW, NW)
        """
        self._check_wind_radii(wind_radii)
        center = (self.bti.shape[0]//2, self.bti.shape[1]//2)
        radii_pixels = [2*radius/self.resolution for radius in wind_radii]
        fig, ax = self.create_plot()
        for i in range(4):
            start_angle = (i-1)*90.
            dx = 0.5*cos(radians(start_angle))
            dy = 0.5*sin(radians(start_angle))
            
            plt.plot((center[1]+dx*radii_pixels[i-1], center[1]+dx*radii_pixels[i]), 
                     (center[0]+dy*radii_pixels[i-1], center[0]+dy*radii_pixels[i]),
                     color='b')
            
            arc = Arc(center, radii_pixels[i], radii_pixels[i], color='b',
                      theta1=start_angle, theta2=start_angle+90)
            ax.add_patch(arc)
        plt.savefig(fn, dpi=100)   
        #plt.show()

