import os
import pyproj
import pandas as pd

from utilities.data_io import get_top_left_corner_coordinates_for_image


path_example_safe = 'path/to/tile'
path_raw_bathy = 'path/to/raw/xyz'

# read raw file
df = pd.read_csv(path_raw_bathy, header=2)
df = df.rename(columns={'long(DD)': 'lng', 'lat(DD)': 'lat', 'depth(m - down positive - LAT)': 'z'})

df = df.sample(10000)
x, y, epsg = get_top_left_corner_coordinates_for_image(path_example_safe)
if epsg != 'EPSG:4326':
    proj = pyproj.Proj(proj='utm', init=epsg, ellps='WGS84')
    df['lng'], df['lat'] = proj(df['lng'].tolist(), df['lat'].tolist())

df.to_csv(path_raw_bathy.replace('.xyz', '.fxyz'))
