"""
Classes used to read information from various sources.

Supports:
    IBTrACS (doi.org/10.25921/82ty-9e16)             # IbtracsReader
    ERA_5 wind (doi.org/10.24381/cds.adbb2d47)       # WindReader
    ERA_5 radiation: (doi.org/10.24381/cds.adbb2d47) # RadiationReader
    GPM_MERGIR (doi.org/10.5067/P4HZB9N27EKU)        # MergirReader
    DAV                                              # DavReader
    SST (doi.org/10.24381/cds.cf608234)              # SstReader
    GOES16                                           # GoesReader

Dependencies:
    numpy
    xarray
    dask
    netCDF4
    scipy

Author: Joshua May (josh.w.may@gmail.com)
"""

from datetime import datetime
import calendar

import numpy as np
import xarray as xr
from collections import defaultdict
from typing import Union, Tuple
from netCDF4 import Dataset

from scipy.interpolate import CubicSpline

class IbtracsReader:
    """
    A class to support reading from an IBTrACS file. Allows searching for TCs
    and converts raw output to a commonly used format.

    Initialisation can be done by using the code: reader = IbtracsReader(ibtracs_fn)

    Attributes
    ----------
    ibtracs_fn : str
        The filename for the IBTrACS file.
    interpolating : bool
        Whether the output is interpolated to include every 30 minutes.
    columns : str or tuple of str
        Which columns from the IBTrACS file are desired. Two presets are 'basic'
        and 'all'. 'basic' will return TCs name, season, time, latitude, longitude,
        and wind speeds. 'all' will return all columns supported.

    Methods
    -------
    read(*tc_identifier):
        Reads the IBTrACS file and returns a dictionary from information gathered.
        Can used either the name and year of the TC eg: reader.read('Darian', 2022)
        Or can read the dataset directly by using the index: eg: reader.read(12345)
    change_settings(interpolating, columns):
        Preferable method of changing IbtracsReader settings rather than accessing
        it directly (checks inputs) or creating a new object (saves time initialising
        search functionality).

    NOTE:
    When searching for TC with name and year, it follows which season the TC
        is in, rather than the real year. Eg: a TC can occur in December 2004,
        but be registered in IBTrACS as occuring in 2005.
    read() can return a list of dictionaries if there is more than one TC
        that matches the description of name and year.
    Missing datapoints are returned as NaNs.

    """
    def __init__(self, ibtracs_fn,
                 interpolating = True, columns="basic"):
        self.ibtracs = Dataset(ibtracs_fn, 'r')
        self.MAX_INDEX = self.ibtracs.dimensions['storm'].size
        self.year_index_ranges = None

        self.interpolating = self._handle_interpolating_input(interpolating)
        self.column_types = {attr: self._get_attr_type(attr) for attr in self.ibtracs.variables}
        self.columns = self._handle_columns_input(columns)
        
        self.time_resolution = 3
        self.tc_index = None
        self.tc_duration = None
        self.tc_timestamps = None
        self.interpolated_duration = None
        self.sid_dict = None

    @staticmethod
    def _handle_interpolating_input(interpolating):
        if isinstance(interpolating, bool):
            return interpolating
        raise ValueError("'interpolating' input must be bool")

    def _handle_columns_input(self, columns):
        if columns == "basic":
            return ("name", "season", "iso_time", "lat", "lon", "usa_wind",
                    "basin")
        if columns == "thorough":
            return ("name", "season", "iso_time", "lat", "lon", "usa_wind",
                    "basin", "nature", "usa_atcf_id", "dist2land", "usa_r34",
                    "usa_r50", "usa_r64")
        if columns == "db": # for postgres database
            return ("sid", "name", "season", "iso_time", "lat", "lon", 
                    "usa_wind", "basin", "nature", "usa_atcf_id", "dist2land", 
                    "usa_r34", "usa_r50", "usa_r64", "track_type")
        if columns == "all":
            return self.ibtracs.variables
        if isinstance(columns, str):
            columns = [columns]
        columns_str_tup = all(isinstance(column, str) for column in columns)
        if not columns_str_tup:
            msg = f"Argument 'columns' must be 'basic', 'thorough', 'all', or \
                a tuple of column names. Got {columns}."
            raise ValueError(msg)
        ibtracs_vars = set(self.ibtracs.variables)
        for var in columns:
            if var not in ibtracs_vars:
                raise KeyError(f"Variable {var!r} is not in IBTrACS dataset.")
        return columns

    def _get_time_stamps(self):
        days_since_ibtracs_start = self.ibtracs['time'][self.tc_index][:self.tc_duration].data
        raw_stamps = (days_since_ibtracs_start*48).astype(int)
        stamps = raw_stamps - raw_stamps[0]
        self.tc_timestamps = stamps

    def _get_attr_type(self, attr):
        dim = len(self.ibtracs[attr].shape)-1
        dtype = self.ibtracs[attr].dtype.descr[0][1]
        super_type = "num" if ("i" in dtype or "f" in dtype) else "str"
        return f"{dim}d {super_type}"

    def _get_tc_duration(self):
        self.tc_duration = int(self.ibtracs['numobs'][self.tc_index])

    def _get_1dstr_data(self, key):
        return "".join(self.ibtracs[key][self.tc_index].compressed().astype(str))

    def _get_2dstr_data(self, key):
        return ["".join(data.compressed().astype(str))
                for data in self.ibtracs[key][self.tc_index][:self.tc_duration]]

    def _get_0dnum_data(self, key):
        return self.ibtracs[key][self.tc_index].item()

    def _get_ndnum_data(self, key):
        array = self.ibtracs[key][self.tc_index][:self.tc_duration]
        return np.where(array.mask, np.nan, array.data)

    def _get_data(self, key):
        key_type = self.column_types[key]
        if key_type == "1d str":
            return self._get_1dstr_data(key)
        if key_type == "2d str":
            return self._get_2dstr_data(key)
        if key_type == "0d num":
            return self._get_0dnum_data(key)
        if key_type in ["1d num", "2d num"]:
            return self._get_ndnum_data(key)
        raise AttributeError(f"Key ({key}) has unsupported type ({key_type}).")

    def _interpolate_times(self, times):
        interp_times = []
        for time in times:
            day, hour = time[:10],  int(time[11:13])
            max_hour = min(24, hour+self.time_resolution)
            for interp_hour in range(hour, max_hour):
                interp_times.append(f"{day} {interp_hour:02}:00:00")
                interp_times.append(f"{day} {interp_hour:02}:30:00")
        interp_times = list(dict.fromkeys(interp_times)) # remove duplicates
        return interp_times[:-(2*self.time_resolution-1)] # removes extrapolated time steps

    def _interpolate_2dstr_data(self, data):
        repss = np.diff(self.tc_timestamps, append=self.interpolated_duration)
        return [item for item, reps in zip(data, repss) for _ in range(reps)]

    # def _interpolate_1dnum_data(self, data):
    #     return np.interp(np.arange(self.interpolated_duration), self.tc_timestamps, data)

    # def _interpolate_2dnum_data(self, data):
    #     interpolated_data = np.empty((self.interpolated_duration, data.shape[1]))
    #     for column in range(data.shape[1]):
    #         interpolated_data[:,column] = self._interpolate_1dnum_data(data[:,column])
    #     return interpolated_data

    def _interpolate_1dnum_data(self, data):
        x = self.tc_timestamps
        x_new = np.arange(self.interpolated_duration)
        data = np.asarray(data, dtype=float)
    
        # Mask valid values
        valid = ~np.isnan(data)
    
        # Need at least 2 valid points for spline
        if np.sum(valid) < 2:
            return np.full_like(x_new, np.nan, dtype=float)
    
        # remove duplicates by taking first occurrence
        x_valid = x[valid]
        y_valid = data[valid]
    
        x_valid, idx = np.unique(x_valid, return_index=True)
        y_valid = y_valid[idx]
    
        # Need at least 2 unique points
        if len(x_valid) < 2:
            return np.full_like(x_new, np.nan, dtype=float)
    
        # Fit spline only on valid samples
        spline = CubicSpline(x_valid, y_valid, bc_type="natural", extrapolate=False)
        result = spline(x_new)
        return result

    def _interpolate_2dnum_data(self, data):
        interpolated_data = np.full((self.interpolated_duration, data.shape[1]), np.nan, dtype=float)
        for col in range(data.shape[1]):
            interpolated_data[:, col] = self._interpolate_1dnum_data(data[:, col])
        return interpolated_data

    def _interpolate_data(self, tc_dict, key):
        data = tc_dict[key]
        key_type = self.column_types[key]
        if key == "iso_time":
            return self._interpolate_times(data)
        if key_type == "1d str":
            return data
        if key_type == "2d str":
            return self._interpolate_2dstr_data(data)
        if key_type == "0d num":
            return data
        if key_type == "1d num":
            return self._interpolate_1dnum_data(data)
        if key_type == "2d num":
            return self._interpolate_2dnum_data(data)
        raise AttributeError(f"Key ({key}) has unsupported type ({key_type}).")

    def _strdata_from_index(self, index, key):
        current_tc_index = self.tc_index
        self.tc_index = index
        data = self._get_1dstr_data(key)
        self.tc_index = current_tc_index
        return data

    def _initialise_year_index_ranges(self):
        ranges = defaultdict(lambda: [self.MAX_INDEX, -1])
        for index in range(self.MAX_INDEX):
            year = int(self.ibtracs["season"][index])
            ranges[year][0] = min(ranges[year][0], index)
            ranges[year][1] = max(ranges[year][1], index)
        self.year_index_ranges = ranges

    def _get_year_index_ranges(self):
        if self.year_index_ranges is None:
            self._initialise_year_index_ranges()
        return self.year_index_ranges

    def _get_index_from_name(self, name, year):
        first, last = self._get_year_index_ranges()[int(year)]
        found_tcs =  [index for index in range(first, last+1)
                if self._strdata_from_index(index, 'name') == name.upper()]
        if len(found_tcs) == 0:
            raise KeyError("There isn't a TC with that name and year in the IBTrACS dataset")
        if len(found_tcs) == 1:
            return found_tcs[0]
        return found_tcs
   
    def _get_index_from_sid(self, sid):
        if self.sid_dict is None:
            self.sid_dict = {self._strdata_from_index(index, "sid"): index
                             for index in range(self.MAX_INDEX)}
        if sid in self.sid_dict:
            return self.sid_dict[sid]
        raise KeyError(f"There isn't a TC with that SID in the IBTrACS dataset ({sid}).")

    def _handle_tc_identifier(self, tc_identifier):
        if len(tc_identifier) == 1:
            tc_identifier = tc_identifier[0]
            is_index = isinstance(tc_identifier, int)
            is_sid = isinstance(tc_identifier, str) and len(tc_identifier) == 13
            is_name_and_year = False
        elif len(tc_identifier) == 2:
            is_index = False
            is_sid = False
            is_name_and_year = isinstance(tc_identifier[0],str) and isinstance(tc_identifier[1], int)
        else:
            raise ValueError("Too many tc_identifications, one or two expected, got {tc_identifier}.")
        
        if is_index:
            return tc_identifier
        if is_sid:
            return self._get_index_from_sid(tc_identifier)
        if is_name_and_year:
            return self._get_index_from_name(*tc_identifier)
           
        msg = f"'tc_identifier' must be an integer (if index in ibtracs dataset) \
, string (if SID) or name and year pair (str and int, respectively). Got \
{tc_identifier!r}."
        raise ValueError(msg)

    def read(self, *tc_identifier: Union[int, Tuple[str, int]]):
        """
        Retrieve a tropical cyclone from the IBTrACS dataset.
    
        Parameters
        ----------
        *tc_identifier : int, str, or (str, int)
            Identifier for the tropical cyclone. Supported formats:

            - int:
                Index of the storm in the IBTrACS dataset.
            - str:
                Storm SID (13-character identifier).
            - (str, int):
                Tuple of (storm name, season year), e.g. ("Fabian", 1988).

        Returns
        -------
        dict or list of dict
            Dictionary containing the requested columns for the storm.

            If multiple storms match a (name, year) query, a list of dictionaries
            is returned.

        Notes
        -----
        - Missing values are represented as NaN.
        - The "season" year is used (not calendar year).
        - Output time resolution is 30 minutes if `interpolating=True`,
          otherwise native IBTrACS resolution (typically 3 hours).
        - All units are as provided in IBTrACS.
          (https://www.ncei.noaa.gov/sites/default/files/2021-07/IBTrACS_v04_column_documentation.pdf)
        - Time is given as ISO strings (UTC).

        Examples
        --------
            ibtracs = IbtracsReader("ibtracs.nc")
            tc = ibtracs.read("Fabian", 1988)
            tc = ibtracs.read("2019273N12345")  # SID
            tc = ibtracs.read(1234)             # index
        """
        self.tc_index = self._handle_tc_identifier(tc_identifier)
        if isinstance(self.tc_index, list):
            return [self.read(index) for index in self.tc_index]
        if self.tc_index >= self.MAX_INDEX:
            msg = f"Index ({self.tc_index}) entered is too large. Max is {self.MAX_INDEX-1}."
            raise IndexError(msg)
        self._get_tc_duration()
        self._get_time_stamps()
        self.interpolated_duration = self.tc_timestamps[-1]+1
        tc_dict = {attribute: self._get_data(attribute) for attribute in self.columns}
        if self.interpolating:
            tc_dict = {key: self._interpolate_data(tc_dict, key) for key in tc_dict.keys()}
        return tc_dict

    def change_settings(self, interpolating=None, columns=None):
        """
        Update reader configuration.

        Parameters
        ----------
        interpolating : bool, optional
            Whether to interpolate data to 30-minute resolution.
        columns : str or sequence of str, optional
            Columns to extract (same options as constructor).

        Notes
        -----
        This avoids reloading the dataset.
        """
        if interpolating is not None:
            self.interpolating = self._handle_interpolating_input(interpolating)
        if columns is not None:
            self.columns = self._handle_columns_input(columns)

    def __repr__(self):
        return  ("IBTrACS reader object.\n"
                 f"Number of TCs stored in loaded IBTrACS: {self.MAX_INDEX:,}.")

class _Chunk:
    """
    Object that selects chunks of images.

    image_shape = shape of the source image
    size = radius around point that is to be taken
    """
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
        """ Returns a tuple of slices that represent the chunk """
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
        """ Move the chunk to the centrepoint 'location' """
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

class _Dimension:
    """ A physical dimension of an image grid """
    def __init__(self, data: xr.Dataset, attr: str, rounding: bool, required_consistent: bool = True):
        self.attribute = attr
        self.values = data[attr]
        if required_consistent:
            steps = self.values.differentiate(coord=attr)
            step_range = steps.max() - steps.min()
            if step_range > 1e-3:
                raise AssertionError(f"Steps for {attr} must be linear.")
        self.x0 = float(self.values[0])
        dx = float(self.values[1]) - self.x0 #Rough value
        self.mx = float(self.values[-1])
        self.min = min(self.x0, self.mx) - abs(dx)/2 # These need to be done manually because the order may be ascending or descending
        self.max = max(self.x0, self.mx) + abs(dx)/2
        self.dx = (self.max-self.min)/len(self.values) # Recompute for added stability
        self.rounding = rounding

    def to_grid(self, coordinate):
        if not (self.min <= coordinate < self.max):
            raise ValueError(f"Dimension {self.attribute!r} got queried "
                             f"{coordinate} when valid range is {self.min} to {self.max}.")
        raw_idx = (coordinate - self.x0)/self.dx
        if self.rounding:
            rounded_value = int(np.floor(raw_idx + 0.5))
            rounded_too_high = rounded_value == len(self)
            if rounded_too_high:
                return rounded_value-1
            return rounded_value
        return raw_idx

    def __len__(self):
        return len(self.values)

class _Reader:
    def __init__(self, files, size: int, data_attribute: str,
                 lon_attribute: str, lat_attribute: str, time_attribute: str, time_consistent: bool = True):
        try:
            self.data =xr.open_mfdataset(files)   
        except ValueError as e:
            msg = "Error reading files. Check all files are standard .nc4 or .nc format."
            raise ValueError(msg) from e

        self.lats  = _Dimension(self.data, lat_attribute, True)
        self.lons  = _Dimension(self.data, lon_attribute, True)
        self.times = _Dimension(self.data, time_attribute, False, time_consistent)

        aspect_ratio = abs(self.lons.dx/self.lats.dx)
        if not 0.999 < aspect_ratio < 1.001:
            raise Exception("Data is not in lat-lon projection.")
        self.chunk = _Chunk(self.data[data_attribute].shape, int(size/self.lons.dx))
        self.MAX_INDEX = len(self.times)

    def iso_to_index(self, iso_time):
        """ Converts iso_time (in format YYYY-MM-DD HH-mm-SS) to index of Reader """
        in_time = datetime.strptime(iso_time, "%Y-%m-%d %H:%M:%S")
        in_unix = calendar.timegm(in_time.timetuple())*1e9
        return self.times.to_grid(in_unix)

    def coordinate_to_location(self, coordinate):
        """ converts (lat, lon) to (y, x) """
        lat, lon = coordinate
        return self.lats.to_grid(lat), self.lons.to_grid(lon)

    def data_from_index(self, attribute, location, index):
        """ Retrieves a chunk of an attribute from a location, at an index (time) """
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
        return np.pad(chunk, padding, constant_values=np.nan)
    
    def read_tc_index(self, tc_dict, index):
        coordinate = (tc_dict['lat'][index], tc_dict['lon'][index])
        return self.read(tc_dict['iso_time'][index], coordinate)

class WindReader(_Reader):
    def __init__(self, wind_files, size: float):
        super().__init__(wind_files, size, 'u', 'longitude', 'latitude', 'time')

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

class MergirReader(_Reader):
    """Class used to interface with the GPM_MERGIR dataset.

    Example:
    -------
        from dav.utils import IbtracsReader, MergirReader

        ibtracs_path = "C:/Users/Public/IBTrACS.ALL.v04r01.nc"
        field_of_view_degrees = 20
        example_time = "2016-06-11 12:30:00"
        example_coordinate = (-37.15, 145.33)

        ibtracs = IbtracsReader(ibtracs_path)
        tc = ibtracs.read("Helene", 2024)

        files = load_files(tc) # dummy function used to identify required MERGIR
        mergir = MergirReader(files, field_of_view_degrees)

        image = mergir.read(example_time, example_coordinate)
        first_tc_image = mergir.read_tc_index(tc, 0)

    Notes:
    -----
         - Opening a large set of images is slow due to the xarray backend.
             It is therefore recommended to open a small set (such as an
             individual tropical cyclone's lifetime, rather than a year's worth
             of images, for example.)
    
    """
    def __init__(self, image_files, size: float):
        """
        image_files = list of files, or string containing radical. Each file should be sequential.
        size = radius (in degrees) from the provided coordinates that will be captured
        """
        super().__init__(image_files, size, 'Tb', 'lon', 'lat', 'time')

    def read(self, iso_time: str, coordinate: tuple):
        """
        iso_time: time in the format of YYYY-MM-DD HH:mm:ss
        coordinate: location in the format of (lat, lon)

        Returns the chunk centered over coordinate. Can raise an AssertionError
        about the index if the time is out of range of the files.

        """
        index = round(self.iso_to_index(iso_time))
        location = self.coordinate_to_location(coordinate)
        return self.data_from_index('Tb', location, round(index))

    def read_tc_index(self, tc_dict, index):
        """
        Read the MERGIR data of a particular index of tc_dict

        Parameters
        ----------
        tc_dict : dict
            dict of a TC. Must contain keys: iso_time, lat, lon.
        index : int
            The index to be read.
        """
        if not isinstance(tc_dict, dict):
            raise ValueError(f"tc_dict must be a dict, got {type(tc_dict)}.")
        valid_keys = all(key in tc_dict for key in ("iso_time", "lat", "lon"))
        if not valid_keys:
            raise ValueError("tc_dict must have iso_time, lat, lon keys. Got"
                             f"{tuple(tc_dict.keys())}")
        return super().read_tc_index(tc_dict, index)

class DavReader(_Reader):
    def __init__(self, image_files, size: float):
        """
        image_files = list of files, or string containing radical. Each file should be sequential.
        size = radius (in degrees) from the provided coordinates that will be captured
        """
        super().__init__(image_files, size, 'DAV', 'lon', 'lat', 'time')

    def read(self, iso_time: str, coordinate: tuple):
        """
        iso_time: time in the format of YYYY-MM-DD HH:mm:ss
        coordinate: location in the format of (lat, lon)

        Returns the chunk centered over coordinate. Can raise an AssertionError
        about the index if the time is out of range of the files.

        """
        index = round(self.iso_to_index(iso_time))
        location = self.coordinate_to_location(coordinate)
        return self.data_from_index('DAV', location, round(index))

class SstReader(_Reader):
    def __init__(self, sst_files):
        super().__init__(sst_files, -1, 'sst', 'longitude', 'latitude', 'valid_time')
    
    def read(self, iso_time: str, coordinate: tuple):
        index = round(self.iso_to_index(iso_time))
        location = self.coordinate_to_location(coordinate)
        return float(self.data['sst'][index, *location])

class RadiationReader(_Reader):
    def __init__(self, radiation_files, size: float):
        super().__init__(radiation_files, size, 'toa_sw_clr_1h', 'lon', 'lat', 'time')
        
    def read(self, iso_time: str, coordinate: tuple, attribute: str):
        index = round(self.iso_to_index(iso_time), ndigits=1)
        location = self.coordinate_to_location(coordinate)
        
        if index%1 == 0.5:
            return 0.5*(self.data_from_index(attribute, location, int(index))
                        +self.data_from_index(attribute, location, int(index)+1))
        if index%1 != 0:
            raise IndexError(f"Got bad index: {index}")
        return self.data_from_index(attribute, location, int(index))
    
    def read_tc_index(self, tc_dict, attribute, index):
        coordinate = (tc_dict['lat'][index], tc_dict['lon'][index])
        return self.read(tc_dict['iso_time'][index], coordinate, attribute)

class GoesReader(_Reader):
    def __init__(self, goes_files, size:float):
        """
        goes_files = list of files, or string containing radical. Each file should be sequential.
        size = radius (in degrees) from the provided coordinates that will be captured
        
        These files must be regridded using Dae's code.
        """
        super().__init__(goes_files, size, 'BT', 'lon', 'lat', 'time', time_consistent=False)
    
    def iso_to_index(self, iso_time, max_mins_threshold=30):
        target = np.datetime64(iso_time)
        time_index = self.data.indexes["time"]
        
        idx = time_index.get_indexer([target], method="nearest")[0]
        nearest = time_index[idx]
        
        delta = abs(nearest - target)
        if delta > np.timedelta64(max_mins_threshold, "m"):
            raise IndexError(f"Time ({iso_time}) is too far from a known value")
        return int(idx)
    
    def read(self, iso_time: str, coordinate: tuple):
        """
        iso_time: time in the format of YYYY-MM-DD HH:mm:ss
        coordinate: location in the format of (lat, lon)
        
        Returns the chunk centered over coordinate. Can raise an AssertionError
        about the index if the time is out of range of the files.
        
        """
        try:
            index = self.iso_to_index(iso_time)
        except IndexError:
            return np.full((2*self.chunk.size, 2*self.chunk.size), np.nan)
        try:
            location = self.coordinate_to_location(coordinate)
        except ValueError:
            return np.full((2*self.chunk.size, 2*self.chunk.size), np.nan)
        location = (self.chunk.y_extent-location[0], location[1])
        return self.data_from_index('BT', location, index)

