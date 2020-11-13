from data_generation.data_io import parse_sentinel2_tiles_metadata, get_bathy_xyz
from data_generation.datasets import create_dataset
from data_generation import config as cfg

# get list of Sentinel-2 Tile objects
tiles = parse_sentinel2_tiles_metadata()
print(tiles)
print(f'len(tiles) = {len(tiles)}')

for tile in tiles:
    print(f'--------------TILE: {tile.id}')
    for safe in tile.safes:
        print(f'safe.corners: {safe.corners}')
        print(f'safe.s2_path: {safe.s2_path}')
        print(f'safe.date: {safe.date}')
        print(f'safe.time: {safe.time}')
        print(f'safe.epsg: {safe.epsg}')
        print(f'safe.tidal_elevation: {safe.tidal_elevation}')

# get bathymetry xyz
bathy_xyz = get_bathy_xyz(tiles)
x, y, z = bathy_xyz
print(len(x), len(y), len(z))
print(x)

# pass params to construct_dataset()
create_dataset(tiles, bathy_xyz, cfg.out_path_csv)
