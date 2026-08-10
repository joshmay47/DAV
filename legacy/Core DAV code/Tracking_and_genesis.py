"""

Created on Fri Feb  2 15:34:33 2024

@author: Josh
"""
from scipy.ndimage import label, binary_erosion, binary_dilation, minimum_filter
import numpy as np
from numba import njit, prange, vectorize
from random import randint
from netCDF4 import Dataset
from DAV_gpu import dav
import matplotlib.pyplot as plt
import glob
import pickle
from global_land_mask.globe import is_land
import cv2
from tqdm import tqdm
import math

class System:
    def __init__(self, lat: float, lon: float, cdav: float, time_start: int):
        self.name = f"{randint(0,1e10):0>10}"
        self.id = self.name
        self.ys = [lat]
        self.xs = [lon]
        self.cdavs = [cdav]
        self.positives = [True]
        self.time_start = time_start
        self._active_system = True
    
    def position(self):
        return (self.ys[-1], self.xs[-1])

    def duration(self):
        return len(self.cdavs)

    def update(self, lat: float, lon: float, cdav: float, positive: bool):
        self.ys.append(lat)
        self.xs.append(lon)
        self.cdavs.append(cdav)
        self.positives.append(positive)
        
    def is_active(self, Ttrack = 48):
        if self._active_system:
            still_active = any(self.positives[-Ttrack:])
            if not still_active:
                self._active_system = False
        return self._active_system
    
    def num_positives(self):
        return sum(self.positives)
    
    def kill_system(self):
        self._active_system = False

class Sky:
    def __init__(self, fn: str, time_index: int, bounds = None, downsample: int = 1):
        self.image_res = 4
        nc4data = Dataset(fn)
        self.time = nc4data['time'][time_index].data

        image = nc4data['Tb'][time_index]
        lats = nc4data['lat'][:].data
        lons = nc4data['lon'][:].data
        
        #self.bti_data = np.where(image.mask, np.nan, image.data)
        self.bti_data = np.where(image.mask, np.nan, image.data)
        self._check_bti_vals()
        
        self._d_lat = lats[1] - lats[0]
        self._d_lon = lons[1] - lons[0]
        self._lat0 = lats[0]
        self._lon0 = lons[0]
        self.iso_time = f"{nc4data.BeginDate} {nc4data.BeginTime[:-5]}"
        
        self._global_image = True
        if not bounds is None: self._crop_image(bounds)
        # if downsample != 1: self._downsample_image(downsample)
        self.to_10km_res(4) # Image resolution is hardcoded for now
        
        self.missing_mask = np.isnan(self.bti_data)
    
    def _check_bti_vals(self):
        """ Checks if the bti is in kelvin by asserting that surface temperature
        values should be between 160 and 335. Record temperatures in infrared. """
        inside_kelvin_range = between_vals(self.bti_data, 160, 335)
        if not inside_kelvin_range.all():
            print(self.bti_data.min(), self.bti_data.max())
            raise ValueError("Found suspicious temperature, ensure that bti is given in Kelvin")
    
    def to_10km_res(self, in_resolution):
        """ Resizes the bti image to be 10km. Raises error if bti is not 2D or
        in_resolution is not a number. """
        if len(self.bti_data.shape) != 2:
            raise ValueError(f"bti must be 2D, got {len(self.bti_data.shape)}D.")
        if not isinstance(in_resolution, (int, float)):
            raise ValueError(f"resolution must be a number, got type {type(in_resolution)}.")
        scaling_factor = in_resolution/10
        out_shape = (int(self.bti_data.shape[1]*scaling_factor), 
                     int(self.bti_data.shape[0]*scaling_factor))
        self._downsample_image(1/scaling_factor)
        self.bti_data = cv2.resize(self.bti_data, out_shape, interpolation = cv2.INTER_LINEAR)

    def get_dav(self, dav_radius = 300):
        dav_data = dav(self.bti_data, dav_radius//self.image_res)
        self.dav_data = np.where(self.missing_mask, np.nan, dav_data)
    
    @staticmethod
    def _bounds_check(bounds):
        assert -60 <= bounds[0] <= 60, f"North bound must be between -60 and 60, got {bounds[0]}"
        assert -60 <= bounds[1] <= 60, f"South bound must be between -60 and 60, got {bounds[1]}"
        assert -180 <= bounds[2] <= 180, f"West bound must be between -180 and 180, got {bounds[2]}"
        assert -180 <= bounds[2] <= 180, f"East bound must be between -180 and 180, got {bounds[3]}"
        assert bounds[0] > bounds[1], f"North bound ({bounds[0]}) must be higher than South bound ({bounds[1]})"
        assert bounds[3] > bounds[2], f"East bound ({bounds[3]}) must be higher than West bound ({bounds[2]})."
    
    def _crop_image(self, bounds):
        self._bounds_check(bounds)
        north, west = self.coordinate_to_location((bounds[0], bounds[2]))
        south, east = self.coordinate_to_location((bounds[1], bounds[3]))
        self.bti_data = self.bti_data[south:north, west:east]
        self._lat0 = bounds[1] + self._d_lat/2
        self._lon0 = bounds[2] + self._d_lon/2
        if (bounds[2] != -180) or (bounds[3] != 180):
            self._global_image = False
    
    def _downsample_image(self, downsample):
        self._lat0 += self._d_lat/downsample
        self._lon0 += self._d_lon/downsample
        self._d_lat *= downsample
        self._d_lon *= downsample
        self.image_res *= downsample
    
    def _calculate_slicings(self, north_border, over_north, south_border, over_south, 
                            west_border, over_west, east_border, over_east,
                            radius_pixels, x_extent, y_extent):
        if over_north:
            slice_v = slice(None, south_border)
        elif over_south:
            slice_v = slice(north_border, None)
        else:
            slice_v = slice(north_border, south_border)
        
        if over_west:
            left_slice_h = slice(west_border+x_extent, None)
            right_slice_h = slice(None, east_border)
        elif over_east:
            left_slice_h = slice(west_border, None)
            right_slice_h = slice(None, east_border-x_extent)
        else:
            left_slice_h = slice(west_border, east_border)
            right_slice_h = left_slice_h
        
        left_slice = (slice_v,left_slice_h)
        right_slice = (slice_v, right_slice_h)
        return left_slice, right_slice
    
    def _generate_mask(self, radius):
        points = np.arange(-round(radius)+0.5, round(radius)+0.5)
        y,x = np.meshgrid(points, points, indexing='ij',sparse=True)
        mask = y**2 + x**2 <= radius**2
        return mask
    
    def _retrieve_pixels(self, data, location, circle_mask, radius_pixels):
        y,x = location
        y_extent = data.shape[0]
        x_extent = data.shape[1]

        if not self._is_valid_integer(x, x_extent):
            raise ValueError(f"x must be an int between 0 and {x_extent}, is {x!r}")
        if not self._is_valid_integer(y, y_extent):
            raise ValueError(f"y must be an int between 0 and {y_extent}, is {y!r}")
        if not self._is_valid_integer(radius_pixels, min(x_extent, y_extent)):
            raise ValueError(f"radius must be an int between 0 and {min(y_extent, x_extent)}, is {radius_pixels!r}")

        north_border = y-radius_pixels
        south_border = y+radius_pixels
        west_border  = x-radius_pixels
        east_border  = x+radius_pixels
        
        over_north = north_border < 0
        over_south = south_border > y_extent
        over_west  = west_border < 0 
        over_east  = east_border > x_extent

        left_slice, right_slice = self._calculate_slicings(north_border, over_north,
                                                           south_border, over_south,
                                                           west_border,  over_west,
                                                           east_border,  over_east,
                                                           radius_pixels, x_extent, y_extent)
        if self._global_image:
            if over_west or over_east:
                chunk = np.hstack((data[left_slice],data[right_slice]))
            else:
                chunk = data[left_slice]# or right_slice, they're equivalent when no clipping
            mask = circle_mask
        else:
            if over_west:
                chunk = data[right_slice]
                mask = circle_mask[:,-west_border:]
            elif over_east:
                chunk = data[left_slice]
                mask = circle_mask[:,:x_extent-east_border]
            else:
                chunk = data[left_slice]
                mask = circle_mask
            
        if over_north:
            mask = mask[-north_border:]
        elif over_south:
            mask = mask[:y_extent-south_border]
        
        return chunk, mask
    
    @staticmethod
    def _is_valid_integer(value, max_value):
        return np.issubdtype(type(value), np.integer) and (0 <= value < max_value)
    
    def average_temperature_around_location(self, location, Rcloud=250):
        radius_pixels = int(Rcloud/self.image_res)
        mask = self._generate_mask(radius_pixels)
        cluster_region, cluster_mask = self._retrieve_pixels(self.bti_data, location, mask, radius_pixels)
        cluster_pixels = cluster_region[cluster_mask]
        average = np.nanmean(cluster_pixels)
        return average
    
    def filter_cool_locations(self, locations, Rcloud=250, THBmin = 273):
        is_cool = [self.average_temperature_around_location(location,
                                                            Rcloud = Rcloud) 
                   <= THBmin for location in locations]
        return np.array(is_cool, dtype=np.bool_)
    
    def location_to_coordinates(self, location):
        """location in the form array(<lat>, <lon>)"""
        lat_coord = self._d_lat*location[0]+self._lat0
        lon_coord = self._d_lon*location[1]+self._lon0
        return np.array((lat_coord, lon_coord))
    
    def coordinate_to_location(self, coordinate):
        """coordinate in the form array(<lat>, <lon>)"""
        lat_loc = np.round((coordinate[0]-self._lat0)/self._d_lat)
        lon_loc = np.round((coordinate[1]-self._lon0)/self._d_lon)
        return np.array((lat_loc, lon_loc), dtype=int)

    def _latitude_to_location(self, latitude):
        """Returns the corresponding coordinate from latitude, cropped between
        0 and the maximum latitude coordinate."""
        coord = (latitude - self._lat0)//self._d_lat
        coord = min(max(coord, 0), len(self.bti_data))
        return coord
    
    def filter_within_bounds(self, locations, max_latitude_magnitude):
        min_lat = self._latitude_to_location(-max_latitude_magnitude)
        max_lat = self._latitude_to_location(max_latitude_magnitude)
        is_within = [min_lat <= location[0] <= max_lat for location in locations]
        return np.array(is_within, dtype=np.bool_)
    
    def is_land(self, location):
        coordinate = self.location_to_coordinates(location)
        return is_land(coordinate[0], coordinate[1])
    
    @staticmethod
    @njit
    def _map_to_locations(filtered_map, original_map, labeled_array, num_clusters):
        min_dav_locations = [(-9999,-9999)] * (num_clusters + 1)
        min_dav_values = [np.inf] * (num_clusters + 1)

        for i in range(filtered_map.shape[0]):
            for j in range(filtered_map.shape[1]):
                if filtered_map[i, j] and original_map[i, j] < min_dav_values[labeled_array[i, j]]:
                    min_dav_values[labeled_array[i, j]] = original_map[i, j]
                    min_dav_locations[labeled_array[i, j]] = (i,j)
        return min_dav_locations[1:]
    
    def find_clusters(self, threshold=2000):
        """
        Parameters
        ----------
        dav_map : 2D float array
            2D IMAGE OF DAV VALUES OF A REGION OF SKY.
        threshold : int or float, optional
            THE HIGHEST DAV VALUE THAT IS CONSIDERED SYMETTRICAL ENOUGH. 
            The default is 2000.
    
        Returns
        -------
        List of tuples
            A LIST OF (X,Y) COORDINATES CORRESPONDING TO CLUSTER LOCATIONS, THESE 
            ARE FILTERED BY THE SIZE OF THE CLUSTER THAT FALLS BELOW TH_MAX.
        """
        below_threshold_map = self.dav_data < threshold
        large_cluster_map = binary_erosion(below_threshold_map,iterations=5)
        return np.array(self._map_to_locations(large_cluster_map,
                                               self.dav_data,
                                               *label(large_cluster_map)))
    
    def min_dav_location_around_origin(self, origin, radius = 150):
        radius_pixels = int(radius/self.image_res)
        mask = self._generate_mask(radius_pixels)
        cluster_region, cluster_mask = self._retrieve_pixels(self.dav_data, origin, mask, radius_pixels)
        cluster_region = np.where(cluster_mask, cluster_region, np.nan)
        all_nan_slice = np.isnan(cluster_region).all()
        if all_nan_slice:
            return origin
        min_y_cluster, min_x_cluster = np.unravel_index(np.nanargmin(cluster_region), cluster_region.shape)
        min_y_global =  origin[0] - radius_pixels + min_y_cluster
        min_x_global = origin[1] - radius_pixels + min_x_cluster
        if self._global_image:
            min_x_global %= self.dav_data.shape[1]
        min_y_global = min(max(0, min_y_global), self.dav_data.shape[0])
        return min_y_global, min_x_global

@vectorize
def between_vals(array, lower, upper):
    if math.isnan(array):
        return True
    return lower <= array <= upper

@njit
def find_distance_squared_matrix(locations, previous_locations):
    distances = np.empty((len(locations), len(previous_locations)), dtype=np.float32)
    for i,(ny, nx) in enumerate(locations):
        for j,(oy, ox) in enumerate(previous_locations):
            distances[i,j] = (nx-ox)**2 + (ny-oy)**2
    return distances

@njit
def minimum_dav_index(dav_values, locations, locations_of_interest_bool):
    min_index = -1
    min_value = 9999.
    for i, (location, interest) in enumerate(zip(locations, locations_of_interest_bool)):
        if not interest:
            continue
        dav_value = dav_values[location[0], location[1]]
        if dav_value < min_value:
            min_value = dav_value
            min_index = i
    return min_index

def identify_independent_locations(locations, image: Sky, closest_distance = 300):
    threshold_squared = (closest_distance/image.image_res)**2
    return identify_independent_locations_aux(locations, image.dav_data, threshold_squared)

@njit
def identify_independent_locations_aux(locations, davs, threshold_squared):
    distance_squared_matrix = find_distance_squared_matrix(locations, locations)
    close_locations = distance_squared_matrix < threshold_squared
    kept_locations = np.ones(len(locations), dtype=np.bool_)
    for row in range(len(locations)):
        locations_of_interest_bool = close_locations[row] & kept_locations
        if np.count_nonzero(locations_of_interest_bool) < 2:
            continue
        kept_location = minimum_dav_index(davs, locations, locations_of_interest_bool)
        kept_locations = kept_locations & ~locations_of_interest_bool
        kept_locations[kept_location] = True
    return kept_locations

def identify_new_locations(old_locations, new_locations, image: Sky, closest_distance = 300):
    threshold_squared = (closest_distance/image.image_res)**2
    distance_squared_matrix = find_distance_squared_matrix(new_locations, old_locations)
    close_locations = distance_squared_matrix < threshold_squared
    new = np.sum(close_locations, axis=0) < 1
    return new

def kill_merging_systems(system_list, image: Sky, closest_distance = 300):
    if len(system_list) < 2: return
    threshold_squared = (closest_distance/image.image_res**2)
    for start, system_1 in enumerate(system_list):
        for end, system_2 in enumerate(system_list[start+1:]):
            both_active = system_1.is_active() and system_2.is_active()
            if not both_active: continue
            pos_1 = system_1.position()
            pos_2 = system_2.position()
            distance_squared = (pos_1[0]-pos_2[0])**2 + (pos_1[1]-pos_2[1])**2
            merging = distance_squared < threshold_squared
            if not merging: continue
            if system_1.num_positives() > system_2.num_positives():
                system_2.kill_system()
            else:
                system_1.kill_system()
            
def update_system_list(system_list, sky: Sky, time, low_dav_val = 2000):
    banned_locations = []
    for system in system_list:
        if system.is_active():
            new_position = sky.min_dav_location_around_origin(system.position())
            moved_out_vertically = (new_position[0] == 0) or (new_position[0] == sky.bti_data.shape[0])
            moved_out_horizontally = (new_position[1] == 0) or (new_position[1] == sky.bti_data.shape[1])
            moved_out = moved_out_vertically or (moved_out_horizontally and not sky._global_image)
            if moved_out:
                system._isactive = False
                continue 
            cold = sky.average_temperature_around_location(new_position) < 273
            dav_value = sky.dav_data[new_position[0], new_position[1]]
            low_dav =  dav_value < low_dav_val
            valid = cold and low_dav
            system.update(*new_position, dav_value, valid)
            banned_locations.append(new_position)
    
    kill_merging_systems(system_list, sky)
    
    clusters = sky.find_clusters(threshold = low_dav_val)
    if clusters.size == 0:
        return
    independent = identify_independent_locations(clusters, sky)
    clusters = clusters[independent]
    if banned_locations != []:
        new = identify_new_locations(clusters, np.array(banned_locations), sky)
        clusters = clusters[new]

    for cluster_loc in clusters:
        if sky.is_land(cluster_loc):
            continue
        warm = sky.average_temperature_around_location(cluster_loc) > 273
        if warm:
            continue
        dav_value = sky.dav_data[cluster_loc[0], cluster_loc[1]]
        new_system = System(*cluster_loc, dav_value, time)
        system_list.append(new_system)

def save_system_plot(skies, system_list, time, fn, 
                     known_system_locs = [],
                     known_system_names = []):
    plt.figure(figsize=(skies[time].bti_data.shape[1]//100, skies[time].bti_data.shape[0]//100))
    plt.imshow(skies[time].bti_data, cmap="gray_r")
    dav_map_mask = binary_dilation(skies[time].bti_data < 273, iterations=10)
    dav_map_mask &=  skies[time].dav_data < 2450
    dav_map = np.where(dav_map_mask, skies[time].dav_data, np.nan)
    plt.imshow(dav_map, alpha=0.3, cmap="turbo")
    
    for system in system_list:
        system_index = time-system.time_start
        if not (0 <= system_index < system.duration()):
            continue
        display_kwargs = {'x': system.xs[system_index], 
                          'y': system.ys[system_index],
                          'c': 'b' if system.positives[system_index] else 'r'}
        plt.scatter(**display_kwargs)
        plt.text(**display_kwargs, s=system.name)
    
    if known_system_locs != []:
        plt.scatter(known_system_locs[:,1], known_system_locs[:,0], c='g')
        for (y, x),name in zip(known_system_locs, known_system_names):
            plt.text(x, y, name, c='g')
    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(fn, dpi=100, bbox_inches = 'tight')
    
def format_ibtracs_times(ibtracs, cut=11300):
    return [[joined for time in storm if (joined := ''.join(time))] 
            for storm in ibtracs['iso_time'][cut:].data.astype(str)]

def search_ibtracs(formatted_ibtracs, iso_time, cut=11300):
    storm_idxs = [i for i, storm in enumerate(formatted_ibtracs, start=cut) 
                 if iso_time in storm]
    return [(storm_idx, formatted_ibtracs[storm_idx-cut].index(iso_time))
            for storm_idx in storm_idxs]

def convert_ibtracs_to(ibtracs, idx, key, dtype):
    return dtype("".join(ibtracs[key][idx].data.astype(str)))

def ibtracs_to_locs(ibtracs, ibtracs_times, skies):
    locs = []
    names = []
    for sky in skies:
        time = sky.iso_time
        three_hourly = int(time[11:13])%3 == 0
        if three_hourly:
            locs_time = []
            names_time = []
            tc_ids = search_ibtracs(ibtracs_times, time)
            for tc_id, t in tc_ids:
                tc_name = convert_ibtracs_to(ibtracs, tc_id, "name", str)
                tc_lat = ibtracs["lat"][tc_id][t]
                tc_lon = ibtracs["lon"][tc_id][t]
                tc_loc = sky.coordinate_to_location((tc_lat, tc_lon))
                locs_time.append(tc_loc)
                names_time.append(tc_name)
        locs.append(locs_time)
        names.append(names_time)
    return np.array(locs), names

def find_systems(fns: str):
  print("Initialising images")
  img_fns = glob.glob("C:/Users/Josh/Documents/Uni/Masters/Data/Satellite/merg_20220901*.nc4")
  skies = []
  for img_fn in tqdm(img_fns):
      for time in range(2):
          img = Sky(img_fn,time, bounds = (40, -40, -180, 180), downsample=2)
          skies.append(img)
  print("Fillings")
  fillings = []
  for i in tqdm(range(1, len(skies)-1)):
      before = np.where(skies[i-1].missing_mask, 0, skies[i-1].bti_data)
      after = np.where(skies[i+1].missing_mask, 0, skies[i+1].bti_data)
      divisor = 2 - skies[i-1].missing_mask.astype(int) - skies[i+1].missing_mask.astype(int)
      filling = (before+after)/divisor
      fillings.append(filling)
  for sky, filling in tqdm(zip(skies[1:-1], fillings)):
      sky.bti_data[sky.missing_mask] = filling[sky.missing_mask]
      sky.missing_mask = sky.missing_mask & np.isnan(filling)
  print("Getting DAVs")
  for i, sky in tqdm(enumerate(skies)):
      sky.get_dav(dav_radius=300)
  
  system_list = []
  print("Analysing systems")
  for time, sky in tqdm(enumerate(skies[1:], 1)):
      update_system_list(system_list, sky, time, low_dav_val=2000)

  return system_list, skies
