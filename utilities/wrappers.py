import os
import gdal
import warnings
import numpy as np
import pandas as pd

from scipy import ndimage
from utilities.data_io import get_tidal_elevation_for_image, get_top_left_corner_coordinates_for_image


class Sentinel2Tile:
    def __init__(self):
        self.id = None
        self.safes = []
        self.epsgs = []
        self.corner = {'x': None, 'y': None}

    def __str__(self):
        return 'Sentinel2Tile:{' \
               f'   ID: {self.id},' \
               f'   SAFES: {self.safes},' \
               f'   EPSG\'S: {self.epsgs}' \
               '}'

    def __eq__(self, other):
        return self.id == other.id


class Sentinel2Safe:
    def __init__(self, safe_path=None, tile_id=None):
        if safe_path:
            self.id = safe_path.split('/')[-1]
            self.tile_id = tile_id
            if not tile_id:
                warnings.warn('NO TILE ID SET IN SAFE CONSTRUCTOR. STOPPING.')
                exit()
            date = self.id[11:19]
            t_time = self.id[20:26]
            self.date = date
            self.time = t_time
            self.s2_path = safe_path
            x, y, epsg = get_top_left_corner_coordinates_for_image(safe_path)
            self.corners = (x, y)
            self.epsg = epsg
            tidal = get_tidal_elevation_for_image(self.id)
            if tidal:
                self.tidal_elevation = tidal
            else:
                warnings.warn(f'No tidal elevation data for safe: {self.id}')
                # exit()
                self.tidal_elevation = 0
        else:
            self.id = None
            self.tile_id = None
            self.corners = None
            self.s2_path = None
            self.date = None
            self.time = None
            self.epsg = None
            self.tidal_elevation = None
        self.l = 109_800

    @property
    def south(self):
        return np.min(self.corners[1] - self.l)

    @property
    def north(self):
        return np.max(self.corners[1])

    @property
    def west(self):
        return np.max(self.corners[0])

    @property
    def east(self):
        return np.min(self.corners[0] + self.l)

    def get_subtile_between_coordinates(self, north, south, east, west):
        cx = (west - self.corners[0]) / 10
        cy = (self.corners[1] - north) / 10
        w, h = int((east - west) / 10), int((north - south) / 10)
        path = os.path.join(self.s2_path, 'GRANULE')
        d = self.id[11:26]
        l1 = os.listdir(path)[0]
        path = os.path.join(path, l1, 'IMG_DATA', f'{self.tile_id}_{d}_')
        tile = np.empty((h, w, 4))
        tile[:, :, 0] = np.array(gdal.Open(path + 'B02.jp2').ReadAsArray(cx, cy, w, h)) / 4096
        tile[:, :, 2] = np.array(gdal.Open(path + 'B03.jp2').ReadAsArray(cx, cy, w, h)) / 4096
        tile[:, :, 3] = np.array(gdal.Open(path + 'B04.jp2').ReadAsArray(cx, cy, w, h)) / 4096
        tile[:, :, 1] = np.array(gdal.Open(path + 'B08.jp2').ReadAsArray(cx, cy, w, h)) / 4096
        return tile

    def get_subtile_around_center(self, lng, lat, subtile_size_in_meters=400, rotation_angle=0):
        if rotation_angle == 0:
            padding = 0
            padoverten = None
        else:
            padding = 90
            padoverten = int(padding/10)
        # add padding to account for rotation
        north = lat + ((subtile_size_in_meters / 2) + padding)
        south = lat - ((subtile_size_in_meters / 2) + padding)
        east = lng + ((subtile_size_in_meters / 2) + padding)
        west = lng - ((subtile_size_in_meters / 2) + padding)

        tile = self.get_subtile_between_coordinates(north, south, east, west)
        if rotation_angle != 0:
            tile = ndimage.rotate(tile, rotation_angle, reshape=False)[padoverten:-padoverten,
                   padoverten:-padoverten, :]
        return tile, north-padding, south+padding, east-padding, west+padding

    def get_full_rgb_image(self):
        path = os.path.join(self.s2_path, 'GRANULE')
        d = self.id[11:26]
        l1 = os.listdir(path)[0]
        path = os.path.join(path, l1, 'IMG_DATA', f'{self.tile_id}_{d}_')
        rgb = np.transpose(gdal.Open(path + 'TCI.jp2').ReadAsArray(), (1, 2, 0))
        return rgb

    def get_image_around_bathy(self, bathy):
        return self.get_subtile_between_coordinates(bathy.north, bathy.south, bathy.east, bathy.west)


class BathymetryXYZ:
    # assuming appropriate coordinate system is already set
    def __init__(self, path=None, sample_n=None):
        if path is not None:
            self.id = path.split('/')[-1]
            df = pd.read_csv(path, header=0)
            if sample_n:
                df = df.sample(sample_n)
            self.lng = np.array(df.lng)
            self.lat = np.array(df.lat)
            self.z = np.array(df.z)

            if 'energy' in df.columns:
                self.energies = np.array(df.energy)
            else:
                self.energies = []
            if 'ratio_blue' in df.columns:
                self.blue_ratios = np.array(df.blue_ratio)
            else:
                self.blue_ratios = []
        else:
            self.id = None
            self.lng = []
            self.lat = []
            self.z = []
            self.energies = []
            self.blue_ratios = []

    @property
    def south(self):
        return np.min(self.lat)

    @property
    def north(self):
        return np.max(self.lat)

    @property
    def east(self):
        return np.max(self.lng)

    @property
    def west(self):
        return np.min(self.lng)

    @property
    def width(self):
        return self.east - self.west  # actual dimensions/resolution

    @property
    def height(self):
        return self.north - self.south  # actual dimensions/resolution

    def add_point(self, lng, lat, z):
        self.lng.append(lng)
        self.lat.append(lat)
        self.z.append(z)

    def merge(self, other):
        self.id = f'merge({self.id}AND{other.id})'
        self.lng = np.append(self.lng, other.lng)
        self.lat = np.append(self.lat, other.lat)
        self.z = np.append(self.z, other.z)

    def sample(self, sample_n):
        sample_indices = np.random.choice(np.arange(len(self.lng)), sample_n)
        self.lng = self.lng[sample_indices]
        self.lat = self.lat[sample_indices]
        self.z = self.z[sample_indices]
        # TODO: handle energies, blue_ratios...

    def __len__(self):
        return len(self.lng)

    def write_xyz_file(self, path):
        df = pd.DataFrame()
        df['lng'] = self.lng
        df['lat'] = self.lat
        df['z'] = self.z
        if self.energies:
            df['energy'] = self.energies
        if self.blue_ratios:
            df['blue_ratio'] = self.blue_ratios
        df.to_csv(path, index=False)
