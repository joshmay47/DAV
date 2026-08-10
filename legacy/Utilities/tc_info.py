# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:11:19 2024

@author: Josh
"""

from collections import defaultdict
from typing import Union, Tuple

import numpy as np
from netCDF4 import Dataset

COLUMN_TYPES = {"sid": "0d str",
                "season": "0d str",
                "name": "0d str",

                "basin": "1d str",
                "iso_time": "1d str",
                "nature": "1d str",

                "lat": "1d num",
                "lon": "1d num",
                "dist2land": "1d num",
                "usa_wind": "1d num",

                "usa_r34": "2d num",
                "usa_r50": "2d num",
                "usa_r64": "2d num",
                }

class IbtracsReader:
    """
    A class to support reading from an IBTrACS file. Allows searching for TCs
    and converts raw output to a commonly used format. Support can be added or
    removed by changing the 'COLUMN_TYPES' dictionary in this file.

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
    def __init__(self, ibtracs_fn = "C:/Users/Josh/Documents/IBTrACS.ALL.v04r00.nc",
                 interpolating = True, columns="basic"):
        self.ibtracs = Dataset(ibtracs_fn, 'r')
        self.MAX_INDEX = self.ibtracs.dimensions['storm'].size
        self.year_index_ranges = None

        self.interpolating = self._handle_interpolating_input(interpolating)
        self.columns = self._handle_columns_input(columns)

        self.tc_index = None
        self.tc_duration = None
        self.tc_timestamps = None
        self.interpolated_duration = None

    @staticmethod
    def _handle_interpolating_input(interpolating):
        if isinstance(interpolating, bool):
            return interpolating
        raise ValueError("'interpolating' input must be bool")

    def _handle_columns_input(self, columns):
        if columns == "basic":
            return ("name", "season", "iso_time", "lat", "lon", "usa_wind")
        if columns == "all":
            return COLUMN_TYPES.keys()
        if isinstance(columns, str):
            columns = [columns]
        columns_str_tup = all(isinstance(column, str) for column in columns)
        if not columns_str_tup:
            msg = f"Argument 'columns' must be 'basic', 'all', or a tuple of column \
                names. Got {columns}."
            raise ValueError(msg)
        ibtracs_vars = list(self.ibtracs.variables)
        for var in columns:
            if var not in ibtracs_vars:
                raise KeyError(f"Variable ({var!r}) is not in IBTrACS dataset.")
            if var not in COLUMN_TYPES:
                raise KeyError(f"Variable ({var!r}) is not supported")
        return columns

    def _get_tc_duration(self):
        self.tc_duration = int(self.ibtracs['numobs'][self.tc_index])

    def _get_0dstr_data(self, key):
        return "".join(self.ibtracs[key][self.tc_index].compressed().astype(str))

    def _get_1dstr_data(self, key):
        return ["".join(data.astype(str))
                for data in self.ibtracs[key][self.tc_index][:self.tc_duration]]

    def _get_num_data(self, key):
        array = self.ibtracs[key][self.tc_index][:self.tc_duration]
        return np.where(array.mask, np.nan, array.data)

    def _get_time_stamps(self):
        days_since_ibtracs_start = self.ibtracs['time'][self.tc_index][:self.tc_duration].data
        raw_stamps = (days_since_ibtracs_start*48).astype(int)
        stamps = raw_stamps - raw_stamps[0]
        self.tc_timestamps = stamps

    def _get_data(self, key):
        key_type = COLUMN_TYPES[key]
        if key_type == "0d str":
            return self._get_0dstr_data(key)
        if key_type == "1d str":
            return self._get_1dstr_data(key)
        if key_type in ("1d num", "2d num"):
            return self._get_num_data(key)
        raise AttributeError(f"Key ({key}) has unsupported type ({key_type}).")

    @staticmethod
    def _interpolate_times(times):
        interp_times = []
        for time in times:
            day, hour = time[:10],  int(time[11:13])
            max_hour = min(24, hour+3)
            for interp_hour in range(hour, max_hour):
                interp_times.append(f"{day} {interp_hour:02}:00:00")
                interp_times.append(f"{day} {interp_hour:02}:30:00")
        interp_times = list(dict.fromkeys(interp_times)) # remove duplicates
        return interp_times[:-5] # removes five extrapolated time steps

    def _interpolate_1dstr_data(self, data):
        repss = np.diff(self.tc_timestamps, append=self.interpolated_duration)
        return [item for item, reps in zip(data, repss) for _ in range(reps)]

    def _interpolate_1dnum_data(self, data):
        return np.interp(np.arange(self.interpolated_duration), self.tc_timestamps, data)

    def _interpolate_2dnum_data(self, data):
        interpolated_data = np.empty((self.interpolated_duration, data.shape[1]))
        for column in range(data.shape[1]):
            interpolated_data[:,column] = self._interpolate_1dnum_data(data[:,column])
        return interpolated_data

    def _interpolate_data(self, tc_dict, key):
        data = tc_dict[key]
        key_type = COLUMN_TYPES[key]
        if key == "iso_time":
            return self._interpolate_times(data)
        if key_type == "0d str":
            return data
        if key_type == "1d str":
            return self._interpolate_1dstr_data(data)
        if key_type == "1d num":
            return self._interpolate_1dnum_data(data)
        if key_type == "2d num":
            return self._interpolate_2dnum_data(data)
        raise AttributeError(f"Key ({key}) has unsupported type ({key_type}).")

    def _strdata_from_index(self, index, key):
        current_tc_index = self.tc_index
        self.tc_index = index
        data = self._get_0dstr_data(key)
        self.tc_index = current_tc_index
        return data

    def _initialise_year_index_ranges(self):
        print("Loading TC Search functionality...")
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

    def _handle_tc_identifier(self, tc_identifier):
        if len(tc_identifier) == 1:
            tc_identifier = tc_identifier[0]
            if isinstance(tc_identifier, int):
                return tc_identifier
        if len(tc_identifier) == 2:
            valid_search = isinstance(tc_identifier[0],str) and isinstance(tc_identifier[1], int)
            if valid_search:
                return self._get_index_from_name(*tc_identifier)
        msg = f"'tc_identifier' must be an integer (if index in ibtracs dataset) \
or name and year pair (str and int, respectively). Got {tc_identifier!r}."
        raise ValueError(msg)

    def read(self, *tc_identifier: Union[int, Tuple[str, int]]):
        """

        Parameters
        ----------
        *tc_identifier : int or tuple
            EITHER THE INDEX IN THE IBTRACS DATASET OR A NAME AND YEAR.

        Returns
        -------
        dict or list of dict
            A DICTIONARY WITH THE COLUMNS SPECIFIED (BY THE READER) OF THE TC.
            THIS CAN BE A LIST OF DICTS IF MORE THAN ONE TC MATCHES THE DESCRIPTION
            OF NAME AND YEAR.

        Example Use
        -----------
        reader = IbtracsReader(ibtracs_fn)
        two_tcs = reader.read("Fabian", 1988)
        tc = reader.read(10000)

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
        """Prefereable method of changing reader settings."""
        if interpolating is not None:
            self.interpolating = self._handle_interpolating_input(interpolating)
        if columns is not None:
            self.columns = self._handle_columns_input(columns)

    def __repr__(self):
        return  ("IBTrACS reader object.\n"
                 f"Number of TCs stored in loaded IBTrACS: {self.MAX_INDEX:,}.")
