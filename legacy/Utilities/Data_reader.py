# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:08:58 2024

@author: Josh
"""

from datetime import datetime
import calendar

import numpy as np
import xarray as xr

class Chunk:
    def __init__(self, image_shape, size):
        self.duration, self.y_extent, self.x_extent = image_shape
        self.size = size

        self.north_border = None
        self.south_border = None
        self.west_border  = None
        self.east_border  = None

        self.over_north = None
        self.over_south = None
        self.over_west  = None
        self.over_east  = None

    def calculate_slicings(self, index):
        if self.north_border is None:
            raise RuntimeError("The chunk has not been given a location, run relocate() first.")
        if not 0 <= index < self.duration:
            raise IndexError(f"Requested time index ({index}) out of range of 0-{self.duration}.")
        if self.over_north:
            slice_v = slice(None, self.south_border)
        elif self.over_south:
            slice_v = slice(self.north_border, None)
        else:
            slice_v = slice(self.north_border, self.south_border)

        if self.over_west:
            left_slice_h = slice(self.west_border+self.x_extent, None)
            right_slice_h = slice(None, self.east_border)
        elif self.over_east:
            left_slice_h = slice(self.west_border, None)
            right_slice_h = slice(None, self.east_border-self.x_extent)
        else:
            left_slice_h = slice(self.west_border, self.east_border)
            right_slice_h = left_slice_h

        left_slice = (index, slice_v,left_slice_h)
        right_slice = (index, slice_v, right_slice_h)
        return left_slice, right_slice

    def relocate(self, location):
        y,x = location
        assert 0 <= x < self.x_extent, f"x must be between 0 and {self.x_extent}, is {x}."
        assert 0 <= y < self.y_extent, f"y must be between 0 and {self.y_extent}, is {y}."

        self.north_border = y - self.size
        self.south_border = y + self.size
        self.west_border  = x - self.size
        self.east_border  = x + self.size

        self.over_north = self.north_border < 0
        self.over_south = self.south_border > self.y_extent
        self.over_west  = self.west_border < 0
        self.over_east  = self.east_border > self.x_extent

class Dimension:
    def __init__(self, data: xr.Dataset, attr: str, rounding: bool):
        self.attribute = attr
        self.values = data[attr]
        steps = self.values.differentiate(coord=attr)
        step_range = steps.max() - steps.min()
        if step_range > 1e-3:
            raise AssertionError(f"Steps for {attr} must be linear.")
        self.x0 = float(self.values[0])
        self.dx = float(self.values[1]) - self.x0
        self.mx = float(self.values[-1])
        self.rounding = rounding

    def to_grid(self, coordinate):
        raw_idx = (coordinate - self.x0)/self.dx
        return round(raw_idx) if self.rounding else raw_idx

    def __len__(self):
        return len(self.values)

class Reader:
    def __init__(self, files, size: int, data_attribute: str,
                 lon_attribute: str, lat_attribute: str):
        self.data = xr.open_mfdataset(files)

        self.lats  = Dimension(self.data, lat_attribute, True)
        self.lons  = Dimension(self.data, lon_attribute, True)
        self.times = Dimension(self.data, 'time', False)

        aspect_ratio = abs(self.lons.dx/self.lats.dx)
        if not 0.999 < aspect_ratio < 1.001:
            raise Exception("Data is not in lat-lon projection.")
        self.chunk = Chunk(self.data[data_attribute].shape, int(size/self.lons.dx))
        self.MAX_INDEX = len(self.times)

    def iso_to_index(self, iso_time):
        in_time = datetime.strptime(iso_time, "%Y-%m-%d %H:%M:%S")
        in_unix = calendar.timegm(in_time.timetuple())*1e9
        return self.times.to_grid(in_unix)

    def coordinate_to_location(self, coordinate):
        lat, lon = coordinate
        return self.lats.to_grid(lat), self.lons.to_grid(lon)

    def data_from_index(self, attribute, location, index):
        assert index < self.MAX_INDEX, f"Index {index} out of range ({self.MAX_INDEX})"
        self.chunk.relocate(location)
        left_slice, right_slice = self.chunk.calculate_slicings(index)
        if self.chunk.over_west or self.chunk.over_east:
            chunk = np.hstack((self.data[attribute][left_slice],
                               self.data[attribute][right_slice]))
        else: # or right_slice, they're equivalent when no clipping
            chunk = self.data[attribute][left_slice] 
        chunk = np.array(chunk)
        padding_size = 2*self.chunk.size - chunk.shape[0]
        if self.chunk.over_north:
            padding = ((padding_size, 0), (0,0))
        elif self.chunk.over_south:
            padding = ((0, padding_size), (0,0))
        else:
            return chunk
        return np.pad(chunk, padding)

class WindReader(Reader):
    def __init__(self, wind_files, size: float):
        super().__init__(wind_files, size, 'u', 'longitude', 'latitude')

    def read(self, iso_time: str, coordinate: tuple, direction: str):
        index = self.iso_to_index(iso_time)
        location = self.coordinate_to_location(coordinate)

        if index%1 == 0.5:
            return 0.5*(self.data_from_index(direction, location, int(index))
                        +self.data_from_index(direction, location, int(index)+1))[::-1]
        if index%1 != 0:
            raise IndexError(f"Got bad index: {index}")
        return self.data_from_index(direction, location, int(index))[::-1]

    def read_tc_index(self, tc_dict, direction, index):
        coordinate = (tc_dict['lat'][index], tc_dict['lon'][index])
        return self.read(tc_dict['iso_time'][index], coordinate, direction)

class MergirReader(Reader):
    def __init__(self, image_files, size: float):
        super().__init__(image_files, size, 'Tb', 'lon', 'lat')

    def read(self, iso_time: str, coordinate: tuple):
        index = round(self.iso_to_index(iso_time))
        location = self.coordinate_to_location(coordinate)
        return self.data_from_index('Tb', location, round(index))

    def read_tc_index(self, tc_dict, index):
        coordinate = (tc_dict['lat'][index], tc_dict['lon'][index])
        return self.read(tc_dict['iso_time'][index], coordinate)
